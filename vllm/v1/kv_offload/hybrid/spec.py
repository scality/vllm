# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload KV to host memory and, straight from the GPU, to an object store.

Loads read from both at once: host memory over each GPU's PCIe link, the
object store over the storage NICs. kv_connector_extra_config holds the
options of CPUOffloadingSpec and ObjOffloadingSpec, and:

    spec_name: "HybridOffloadingSpec"
    obj_load_share: if set, the fraction of the chunks of a load read from
        the object store when host memory holds them too. By default the
        split follows the load times measured so far (see split.py),
        starting from:
    host_load_gbps, host_load_latency_ms: host memory's rate of KV bytes
        delivered to all workers and its latency per load (default 40, 1)
    obj_load_gbps, obj_load_latency_ms: the object store's (default 20, 10)
    load_time_half_life_s: how fast older loads stop counting (default 300);
        with no loads for a while the estimates return to the above

With more than one worker, set canonical_layout: every worker then stores,
as the object store needs.
"""

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.hybrid.manager import HybridOffloadingManager
from vllm.v1.kv_offload.hybrid.split import LoadTimeModel
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager
from vllm.v1.kv_offload.obj.spec import ObjOffloadingSpec


class HybridOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        if config.parallel.world_size > 1 and not config.canonical_layout:
            raise ValueError(
                "HybridOffloadingSpec needs canonical_layout with several workers"
            )
        # Host memory serves this instance only: its pages need not be
        # canonical. Replicated KV keeps them canonical, which costs one copy
        # per layer and lets each worker write a share of every chunk;
        # sharded KV would cost a copy per token and head shard.
        self.canonical_layout = config.canonical_layout and self.replicated_layout
        self._obj = ObjOffloadingSpec(config)
        share = self.extra_config.get("obj_load_share")
        self.obj_load_share = None if share is None else float(share)
        if self.obj_load_share is not None and not 0.0 <= self.obj_load_share <= 1.0:
            raise ValueError(
                f"obj_load_share must be in [0, 1], got {self.obj_load_share}"
            )
        half_life = float(self.extra_config.get("load_time_half_life_s", 300))
        self._load_time_priors = {
            medium: dict(
                latency=float(self.extra_config.get(f"{medium}_load_latency_ms", ms))
                / 1e3,
                per_chunk=self.kv_bytes_per_chunk
                / (float(self.extra_config.get(f"{medium}_load_gbps", gbps)) * 1e9),
                half_life=half_life,
            )
            for medium, gbps, ms in (("host", 40, 1), ("obj", 20, 10))
        }
        self._hybrid_manager: HybridOffloadingManager | None = None
        self._hybrid_worker: OffloadingWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._hybrid_manager is None:
            cpu = super().get_manager()
            obj = self._obj.get_manager()
            assert isinstance(cpu, CPUOffloadingManager)
            assert isinstance(obj, ObjOffloadingManager)
            self._hybrid_manager = HybridOffloadingManager(
                cpu,
                obj,
                LoadTimeModel(**self._load_time_priors["host"]),
                LoadTimeModel(**self._load_time_priors["obj"]),
                self.obj_load_share,
            )
        return self._hybrid_manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._hybrid_worker is None:
            from vllm.v1.kv_offload.hybrid.worker import HybridOffloadingWorker

            obj = self._obj.get_worker(kv_caches)
            self._hybrid_worker = HybridOffloadingWorker(
                super().get_worker(kv_caches), obj, self.blocks_per_chunk
            )
        return self._hybrid_worker
