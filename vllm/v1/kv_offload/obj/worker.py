# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of offloading KV straight from the GPU to an object store.

A store copies the chunks' KV into slots of a staging buffer in GPU memory, in
the CPU tier's chunk layout, then PUTs each slot as one object over GPUDirect
RDMA. A load GETs the objects into slots, then copies them into the KV cache.
A job larger than the staging buffer goes through it in pieces. A driver
thread moves pieces from one stage to the next as soon as each completes.

With several workers holding identical KV (replicated layout), each chunk has
one owner worker, which moves it alone: a store PUTs the owner's copy, a load
writes the chunk into every worker's KV cache through CUDA IPC.
"""

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    DevicePointers,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    _canonical_block_sizes,
)
from vllm.v1.kv_offload.obj.client import NIXL_READ, NIXL_WRITE, ObjStoreClient
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec, owner_of
from vllm.v1.kv_offload.obj.peers import AllGather, CrossRankFence, share_tensors
from vllm.v1.kv_offload.pointer_utils import (
    replicate_device_pointers,
    select_device_chunks,
)

logger = init_logger(__name__)

# How long the driver waits between polls while transfers are in flight.
_POLL_INTERVAL_S = 1e-4
# How long a load waits for the other workers to receive it.
_FENCE_TIMEOUT_S = 30.0


class StagingSlots(BlockIDsLoadStoreSpec):
    """Slots of the staging buffer, one per chunk."""


def _sole_writer(ref: CanonicalKVCacheRef) -> CanonicalKVCacheRef:
    if ref.mapping is None or ref.mapping.num_writers == 1:
        return ref
    mapping = dataclasses.replace(ref.mapping, num_writers=1, writer_index=0)
    return dataclasses.replace(ref, mapping=mapping)


@dataclass(eq=False)
class _Job:
    job_id: int
    is_store: bool
    device_ptrs: DevicePointers
    obj_names: list[str]
    # Recorded on the compute stream at submission: a store reads KV written
    # before it, a load writes blocks that may still be being zeroed.
    ready: torch.Event
    submit_time: float
    # Chunks this worker moves, indices into obj_names.
    chunks: list[int]
    next_chunk: int = 0
    in_flight: int = 0
    # Store: pieces being copied out of the KV cache.
    copying: int = 0
    failed: bool = False

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def started(self) -> bool:
        return self.failed or self.next_chunk == self.num_chunks


@dataclass(eq=False)
class _Piece:
    job: _Job
    # Indices into job.obj_names.
    chunks: list[int]
    slots: list[int]

    @property
    def obj_names(self) -> list[str]:
        return [self.job.obj_names[c] for c in self.chunks]


class ObjOffloadingWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        chunk_bytes: int,
        page_offset: int,
        canonical_layout: bool,
        store_config: dict,
        io_threads: int,
        staging_bytes: int,
        agent_name: str,
        rank: int = 0,
        world_size: int = 1,
        all_gather: AllGather | None = None,
    ):
        """Args:
        kv_caches: Canonicalized KV caches.
        blocks_per_chunk: GPU blocks per chunk.
        chunk_bytes: Bytes of one object: a CPU tier chunk row, padding
            included.
        page_offset: Offset of this worker's pages in a chunk, as in the CPU
            tier (direct layout only).
        canonical_layout: Chunks use the canonical layout.
        store_config: Object store parameters (see ObjStoreConfig).
        io_threads: NIXL I/O threads.
        staging_bytes: Size of the staging buffer.
        agent_name: NIXL agent name.
        rank: This worker's rank.
        world_size: Workers holding identical KV; more than one requires the
            replicated layout.
        all_gather: Exchanges objects between the workers, required with
            more than one.

        """
        self._blocks_per_chunk = blocks_per_chunk
        self._chunk_bytes = chunk_bytes
        self._rank = rank
        self._world_size = world_size
        num_slots = staging_bytes // chunk_bytes
        if num_slots < 1:
            raise ValueError(
                f"staging_bytes ({staging_bytes}) must hold at least one chunk "
                f"({chunk_bytes} bytes)"
            )
        self._max_piece_chunks = max(1, num_slots // 4)
        self._device = kv_caches.tensors[0].tensor.device

        self._staging = torch.empty(
            (num_slots, chunk_bytes), dtype=torch.int8, device=self._device
        )
        staging_tensors = self._build_views(
            kv_caches, blocks_per_chunk, page_offset, canonical_layout
        )
        # A chunk's owner moves all of it: no writer rotation between workers.
        refs = [
            [_sole_writer(ref) for ref in group_refs]
            for group_refs in kv_caches.group_data_refs
        ]
        self._store_copies = SingleDirectionOffloadingHandler(
            cpu_tensors=staging_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )
        self._fence: CrossRankFence | None = None
        if world_size > 1:
            assert all_gather is not None
            # A load writes every worker's copy: the same copies, once per
            # worker's tensors, through pointers into each worker's tensors.
            gpu_tensors = [
                t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
                for t in kv_caches.tensors
            ]
            all_tensors = share_tensors(gpu_tensors, rank, all_gather)
            num_tensors = len(gpu_tensors)
            self._own_bases = [t.tensor.data_ptr() for t in kv_caches.tensors]
            self._peer_bases = [
                [t.data_ptr() for t in tensors] for tensors in all_tensors
            ]
            self._load_ref_tensors = [
                [ref.tensor_idx for ref in group_refs] for group_refs in refs
            ]
            staging_tensors = staging_tensors * world_size
            refs = [
                [
                    dataclasses.replace(
                        ref, tensor_idx=ref.tensor_idx + r * num_tensors
                    )
                    for r in range(world_size)
                    for ref in group_refs
                ]
                for group_refs in refs
            ]
            self._fence = CrossRankFence(rank, world_size, all_gather)
        self._load_copies = SingleDirectionOffloadingHandler(
            cpu_tensors=staging_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
        )

        self._client = ObjStoreClient(agent_name, store_config, io_threads)
        try:
            self._client.register_local(
                base_addr=self._staging.data_ptr(),
                num_slots=num_slots,
                slot_bytes=chunk_bytes,
                mem_type="VRAM",
                dev_id=self._device.index or 0,
            )
        except Exception:
            self._client.shutdown()
            raise
        logger.info(
            "Object store staging: %d slots of %d bytes on %s",
            num_slots,
            chunk_bytes,
            self._device,
        )

        # Everything below is guarded by _cond.
        self._cond = threading.Condition()
        self._free_slots: list[int] = list(range(num_slots))
        self._jobs: dict[int, _Job] = {}
        self._waiting_loads: deque[_Job] = deque()
        self._waiting_stores: deque[_Job] = deque()
        self._pieces: dict[int, _Piece] = {}
        self._next_piece_id = 0
        self._results: list[TransferResult] = []
        self._stop = False
        # Set when the driver died: every job fails from then on.
        self._broken = False

        self._stream = current_platform.Stream(device=self._device)
        self._thread = threading.Thread(
            target=self._run, name="vllm_offloading_obj_driver", daemon=True
        )
        self._thread.start()

    def _build_views(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        page_offset: int,
        canonical_layout: bool,
    ) -> list[torch.Tensor]:
        """Views of the staging buffer, laid out like a CPU tier chunk row (see
        SharedOffloadRegion)."""
        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )
        offset = 0 if canonical_layout else page_offset
        staging_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            page_size = kv_cache_tensor.page_size_bytes
            if canonical_bytes_per_block is not None:
                chunk_page_size = canonical_bytes_per_block[t_idx] * blocks_per_chunk
            else:
                chunk_page_size = page_size * blocks_per_chunk
            assert offset + chunk_page_size <= self._chunk_bytes
            staging_tensors.append(
                self._staging.as_strided(
                    (self._staging.shape[0], chunk_page_size),
                    (self._chunk_bytes, 1),
                    storage_offset=offset,
                )
            )
            offset += chunk_page_size
        return staging_tensors

    # Engine thread

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, ObjLoadStoreSpec)
        return self._submit(job_id, True, device_ptrs, dst_spec.obj_names)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        device_ptrs: DevicePointers,
        expedited: bool = False,
    ) -> bool:
        assert isinstance(src_spec, ObjLoadStoreSpec)
        return self._submit(
            job_id, False, device_ptrs, src_spec.obj_names, expedited=expedited
        )

    def _submit(
        self,
        job_id: int,
        is_store: bool,
        device_ptrs: DevicePointers,
        obj_names: list[str],
        expedited: bool = False,
    ) -> bool:
        ready = torch.Event()
        ready.record()
        if not is_store and self._fence is not None:
            self._fence.publish(job_id)
        chunks = [
            i
            for i, name in enumerate(obj_names)
            if owner_of(name, self._world_size) == self._rank
        ]
        job = _Job(
            job_id, is_store, device_ptrs, obj_names, ready, time.perf_counter(), chunks
        )
        with self._cond:
            if not chunks or self._broken:
                self._results.append(TransferResult(job_id, not self._broken))
                return True
            self._jobs[job_id] = job
            if is_store:
                self._waiting_stores.append(job)
            elif expedited:
                # Waited on within the step: ahead of regular loads.
                self._waiting_loads.appendleft(job)
            else:
                self._waiting_loads.append(job)
            self._cond.notify_all()
        return True

    def get_finished(self) -> list[TransferResult]:
        with self._cond:
            results, self._results = self._results, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Wait until these stores have read the KV cache, and these loads have
        completed."""

        def settled(job: _Job | None) -> bool:
            if job is None:
                return True
            return job.is_store and job.started and job.copying == 0

        with self._cond:
            self._cond.wait_for(
                lambda: all(settled(self._jobs.get(j)) for j in job_ids)
            )

    def shutdown(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join()
        try:
            self._client.drain()
            self._stream.synchronize()
            self._store_copies.shutdown()
            self._load_copies.shutdown()
        finally:
            self._client.shutdown()
            if self._fence is not None:
                self._fence.close()

    # Driver thread

    def _run(self) -> None:
        torch.accelerator.set_device_index(self._device)
        with self._cond:
            while not self._stop:
                try:
                    self._step()
                except Exception:
                    logger.exception("Object store driver failed")
                    self._fail_all()
                    return
                if self._pieces:
                    self._cond.wait(_POLL_INTERVAL_S)
                elif not (self._waiting_loads or self._waiting_stores):
                    self._cond.wait()

    def _step(self) -> None:
        for result in self._store_copies.get_finished():
            piece = self._pieces[result.job_id]
            piece.job.copying -= 1
            if not result.success:
                self._fail_piece(result.job_id, "copy out of the KV cache")
                continue
            self._client.submit(
                result.job_id,
                np.asarray(piece.slots, dtype=np.int32),
                piece.obj_names,
                NIXL_WRITE,
            )
            self._cond.notify_all()
        for result in self._load_copies.get_finished():
            if not result.success:
                self._fail_piece(result.job_id, "copy into the KV cache")
                continue
            self._finish_piece(result.job_id)
        for obj_result in self._client.poll():
            piece = self._pieces[obj_result.job_id]
            if not obj_result.success:
                if not piece.job.failed:
                    logger.warning(
                        "Object store %s of job %d failed",
                        "store" if piece.job.is_store else "load",
                        piece.job.job_id,
                    )
                piece.job.failed = True
            if piece.job.is_store or piece.job.failed:
                self._finish_piece(obj_result.job_id)
            else:
                self._copy(obj_result.job_id, piece)
        self._start_pieces(self._waiting_loads)
        self._start_pieces(self._waiting_stores)

    def _start_pieces(self, waiting: deque[_Job]) -> None:
        while waiting and self._free_slots:
            job = waiting[0]
            if job.started:
                waiting.popleft()
                continue
            num_chunks = min(
                job.num_chunks - job.next_chunk,
                len(self._free_slots),
                self._max_piece_chunks,
            )
            slots = self._free_slots[-num_chunks:]
            del self._free_slots[-num_chunks:]
            end = job.next_chunk + num_chunks
            piece = _Piece(job, job.chunks[job.next_chunk : end], slots)
            job.next_chunk = end
            job.in_flight += 1
            piece_id = self._next_piece_id
            self._next_piece_id += 1
            self._pieces[piece_id] = piece
            if job.is_store:
                self._copy(piece_id, piece)
            else:
                self._client.submit(
                    piece_id,
                    np.asarray(slots, dtype=np.int32),
                    piece.obj_names,
                    NIXL_READ,
                )

    def _copy(self, piece_id: int, piece: _Piece) -> None:
        """Copy a piece between the KV cache and its slots."""
        job = piece.job
        device_ptrs = select_device_chunks(
            job.device_ptrs, self._blocks_per_chunk, piece.chunks
        )
        slots = StagingSlots(piece.slots)
        with current_platform.stream(self._stream):
            self._stream.wait_event(job.ready)
            if job.is_store:
                job.copying += 1
                self._store_copies.transfer_async(piece_id, device_ptrs, slots)
            else:
                if self._fence is not None:
                    # Workers use the same block ids: the handler writes them
                    # in every worker's tensors.
                    self._fence.wait(job.job_id, self._stream, _FENCE_TIMEOUT_S)
                    device_ptrs = replicate_device_pointers(
                        device_ptrs,
                        self._load_ref_tensors,
                        self._own_bases,
                        self._peer_bases,
                    )
                self._load_copies.transfer_async(piece_id, device_ptrs, slots)

    def _fail_piece(self, piece_id: int, what: str) -> None:
        job = self._pieces[piece_id].job
        if not job.failed:
            logger.warning("Object store job %d failed: %s", job.job_id, what)
        job.failed = True
        self._finish_piece(piece_id)

    def _finish_piece(self, piece_id: int) -> None:
        piece = self._pieces.pop(piece_id)
        self._free_slots.extend(piece.slots)
        job = piece.job
        job.in_flight -= 1
        if job.in_flight == 0 and job.started:
            del self._jobs[job.job_id]
            self._results.append(
                TransferResult(
                    job_id=job.job_id,
                    success=not job.failed,
                    transfer_size=len(job.chunks) * self._chunk_bytes,
                    transfer_time=time.perf_counter() - job.submit_time,
                )
            )
        self._cond.notify_all()

    def _fail_all(self) -> None:
        self._broken = True
        for job in self._jobs.values():
            self._results.append(TransferResult(job.job_id, False))
        self._jobs.clear()
        self._waiting_loads.clear()
        self._waiting_stores.clear()
        self._pieces.clear()
        self._cond.notify_all()
