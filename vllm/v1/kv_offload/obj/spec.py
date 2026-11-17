# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload KV straight from the GPU to an object store, without a CPU tier.

Objects have the names and bytes of the obj secondary tier's, so either path
reads what the other wrote. kv_connector_extra_config:

    spec_name: "ObjOffloadingSpec"
    store_config: object store parameters (see ObjStoreConfig)
    prefix: object name prefix, as for the obj secondary tier
    io_threads: NIXL I/O threads (default 4)
    staging_bytes: GPU staging buffer per worker (default 256 MiB)
    max_inflight_store_chunks: chunks being stored above which new stores
        wait (default 4096)
    reuse_notify: see ReuseNotifier.from_config

Set kv_load_failure_policy to "recompute": a load fails when the store has
evicted the chunk since it was looked up. With more than one worker, chunks
must not depend on the parallel configuration (see canonical_layout); each
chunk then has one owner worker, which moves it for all: it gathers a chunk
sharded across the workers through CUDA IPC, or stores its own copy of a
replicated one.
"""

from typing import Any

from typing_extensions import override

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.obj.client import ObjStoreClient
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager


def _tp_all_gather(obj: Any) -> list[Any]:
    import torch.distributed

    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    gathered: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, obj, group=group.cpu_group)
    return gathered


class ObjOffloadingSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        world_size = config.parallel.world_size
        if world_size != 1 and not (
            config.replicated_layout or config.parallel.is_parallelism_agnostic
        ):
            raise NotImplementedError(
                "ObjOffloadingSpec needs KV whose chunks do not depend on the "
                f"parallel configuration with several workers (world_size="
                f"{world_size}); set canonical_layout"
            )
        store_config: dict[str, Any] | None = self.extra_config.get("store_config")
        if not store_config:
            raise ValueError("store_config must be set in kv_connector_extra_config")
        self.store_config = store_config
        self.prefix: str = self.extra_config.get("prefix", "")
        self.io_threads = int(self.extra_config.get("io_threads", 4))
        self.staging_bytes = int(self.extra_config.get("staging_bytes", 256 * 1024**2))
        self.max_inflight_store_chunks = int(
            self.extra_config.get("max_inflight_store_chunks", 4096)
        )
        self.replicated_layout = config.replicated_layout

        # One object is one CPU tier chunk row (see CPUOffloadingSpec).
        num_copies = 1 if self.replicated_layout else world_size
        kv_bytes_per_chunk = (
            config.worker_kv_bytes_per_block * num_copies * self.blocks_per_chunk
        )
        self.page_size_per_worker = kv_bytes_per_chunk // num_copies
        self.chunk_bytes = round_up(
            kv_bytes_per_chunk, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
        )

        self._manager: OffloadingManager | None = None
        self._worker: OffloadingWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            root_dir = f"{self.prefix}/" if self.prefix else ""
            file_mapper = FileMapper.from_offloading_spec(
                root_dir, self, parallel_agnostic=True
            )
            self._manager = ObjOffloadingManager(
                file_mapper=file_mapper,
                client=ObjStoreClient(
                    "ObjSchedulerAgent", self.store_config, self.io_threads
                ),
                max_inflight_store_chunks=self.max_inflight_store_chunks,
                reuse_notify=self.extra_config.get("reuse_notify"),
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            from vllm.v1.kv_offload.obj.worker import ObjOffloadingWorker

            canonical_layout = self.config.canonical_layout
            if canonical_layout and any(
                ref.mapping is None
                for group_refs in kv_caches.group_data_refs
                for ref in group_refs
            ):
                raise RuntimeError(
                    "canonical_layout was requested but the KV cache layout "
                    "could not be certified for canonical offload."
                )
            rank = self.config.parallel.rank
            world_size = self.config.parallel.world_size
            self._worker = ObjOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                chunk_bytes=self.chunk_bytes,
                rank_page_bytes=0
                if self.replicated_layout or canonical_layout
                else self.page_size_per_worker,
                canonical_layout=canonical_layout,
                store_config=self.store_config,
                io_threads=self.io_threads,
                staging_bytes=self.staging_bytes,
                agent_name=f"ObjWorkerAgent{rank}",
                rank=rank,
                world_size=world_size,
                all_gather=_tp_all_gather if world_size > 1 else None,
                replicated=self.replicated_layout,
            )
        return self._worker
