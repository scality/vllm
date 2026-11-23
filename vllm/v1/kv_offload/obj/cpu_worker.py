# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of offloading KV straight from a CPU KV cache to an object store.

The CPU counterpart of ObjOffloadingWorker, for one worker and the direct
layout. A store copies the chunks' KV into slots of a staging buffer in host
memory, in the CPU tier's chunk layout, then PUTs each slot as one object over
RDMA. A load GETs the objects into slots, then copies them into the KV cache.
A job larger than the staging buffer goes through it in pieces. A driver
thread runs the copies, which release the GIL, and moves pieces from one stage
to the next.
"""

import ctypes
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    DevicePointers,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs
from vllm.v1.kv_offload.obj.client import NIXL_READ, NIXL_WRITE, ObjStoreClient
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.pointer_utils import (
    iter_pointer_groups,
    select_device_chunks,
)

logger = init_logger(__name__)


@dataclass
class _Job:
    job_id: int
    is_store: bool
    device_ptrs: DevicePointers
    obj_names: list[str]
    submit_time: float
    next_chunk: int = 0
    # Pieces not finished yet.
    in_flight: int = 0
    # Store pieces still copying out of the KV cache.
    copying: int = 0
    failed: bool = False

    @property
    def num_chunks(self) -> int:
        return len(self.obj_names)

    @property
    def started(self) -> bool:
        """Whether every chunk is in a piece."""
        return self.next_chunk == self.num_chunks


@dataclass
class _Piece:
    job: _Job
    chunks: list[int]
    slots: list[int] = field(default_factory=list)

    @property
    def obj_names(self) -> list[str]:
        return [self.job.obj_names[i] for i in self.chunks]


class _CopyOps:
    """(KV cache pointer, staging pointer, size) of every page a transfer moves,
    in the order ObjOffloadingWorker's copy handlers lay a chunk row out."""

    def __init__(
        self,
        staging: torch.Tensor,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
    ):
        self._blocks_per_chunk = blocks_per_chunk
        self._refs = kv_caches.group_data_refs
        chunk_bytes = staging.shape[1]
        # A chunk row holds each tensor's pages for the chunk, tensor after
        # tensor (see SharedOffloadRegion).
        used = sorted({ref.tensor_idx for group in self._refs for ref in group})
        self._staging_views: dict[int, torch.Tensor] = {}
        offset = 0
        for t_idx in used:
            page_size = kv_caches.tensors[t_idx].page_size_bytes * blocks_per_chunk
            assert offset + page_size <= chunk_bytes
            self._staging_views[t_idx] = staging.as_strided(
                (staging.shape[0], page_size), (chunk_bytes, 1), storage_offset=offset
            )
            offset += page_size

    def build(
        self, device_ptrs: DevicePointers, slots: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kv_ptrs: list[np.ndarray] = []
        staging_ptrs: list[np.ndarray] = []
        sizes: list[np.ndarray] = []
        slot_ids = np.asarray(slots, dtype=np.int64)
        slot_offset = 0
        for g in iter_pointer_groups(device_ptrs, self._blocks_per_chunk):
            group_slots = slot_ids[slot_offset : slot_offset + g.n_chunks]
            for d, ref in enumerate(self._refs[g.group_idx]):
                start = g.dev_ptr_offset + d * g.group_size
                kv_ptrs.append(device_ptrs.ptrs[start : start + g.group_size])
                out = np.empty(g.group_size, dtype=np.uint64)
                compute_sub_block_ptrs(
                    group_slots,
                    self._blocks_per_chunk,
                    out,
                    self._staging_views[ref.tensor_idx],
                    skip_count=g.skip,
                )
                staging_ptrs.append(out)
                sizes.append(np.full(g.group_size, ref.page_size_bytes, np.uint64))
            slot_offset += g.n_chunks
        assert slot_offset == len(slots)
        if not kv_ptrs:
            empty = np.empty(0, dtype=np.uint64)
            return empty, empty, empty
        return (
            np.concatenate(kv_ptrs),
            np.concatenate(staging_ptrs),
            np.concatenate(sizes),
        )


def _memmove_all(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray) -> None:
    """Copy each range, merging ranges contiguous on both sides first."""
    if len(sizes) == 0:
        return
    breaks = np.flatnonzero(
        (dst[1:] != dst[:-1] + sizes[:-1]) | (src[1:] != src[:-1] + sizes[:-1])
    )
    starts = np.concatenate(([0], breaks + 1))
    run_bytes = np.add.reduceat(sizes, starts)
    for s, n in zip(starts.tolist(), run_bytes.tolist()):
        ctypes.memmove(int(dst[s]), int(src[s]), n)


class ObjCpuOffloadingWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        chunk_bytes: int,
        store_config: dict,
        io_threads: int,
        staging_bytes: int,
        agent_name: str,
        piece_chunks: int | None = None,
        max_inflight_gets: int | None = None,
    ):
        """Args:
        kv_caches: KV caches in host memory, direct layout.
        blocks_per_chunk: KV cache blocks per chunk.
        chunk_bytes: Bytes of one object: a CPU tier chunk row, padding
            included.
        store_config: Object store parameters (see ObjStoreConfig).
        io_threads: NIXL I/O threads.
        staging_bytes: Size of the staging buffer.
        agent_name: NIXL agent name.
        piece_chunks: Most chunks a piece moves through the staging buffer
            (default a quarter of its slots).
        max_inflight_gets: Most chunks being read from the store at once
            (default all the slots).

        """
        self._blocks_per_chunk = blocks_per_chunk
        self._chunk_bytes = chunk_bytes
        num_slots = staging_bytes // chunk_bytes
        if num_slots < 1:
            raise ValueError(
                f"staging_bytes ({staging_bytes}) must hold at least one chunk "
                f"({chunk_bytes} bytes)"
            )
        self._max_piece_chunks = piece_chunks or max(1, num_slots // 4)
        self._max_inflight_gets = max_inflight_gets or num_slots

        # Zeroed: a row's padding is never written, so objects are
        # deterministic.
        self._staging = torch.zeros((num_slots, chunk_bytes), dtype=torch.int8)
        self._ops = _CopyOps(self._staging, kv_caches, blocks_per_chunk)

        self._client = ObjStoreClient(agent_name, store_config, io_threads)
        try:
            self._client.register_local(
                base_addr=self._staging.data_ptr(),
                num_slots=num_slots,
                slot_bytes=chunk_bytes,
                mem_type="DRAM",
            )
        except Exception:
            self._client.shutdown()
            raise
        logger.info(
            "Object store staging: %d slots of %d bytes in host memory",
            num_slots,
            chunk_bytes,
        )

        # Everything below is guarded by _cond.
        self._cond = threading.Condition()
        self._free_slots: list[int] = list(range(num_slots))
        self._jobs: dict[int, _Job] = {}
        self._waiting_loads: deque[_Job] = deque()
        self._waiting_stores: deque[_Job] = deque()
        self._pieces: dict[int, _Piece] = {}
        self._next_piece_id = 0
        self._gets_in_flight = 0
        self._results: list[TransferResult] = []
        self._stop = False
        # Set when the driver died: every job fails from then on.
        self._broken = False

        self._thread = threading.Thread(
            target=self._run, name="vllm_offloading_obj_cpu_driver", daemon=True
        )
        self._thread.start()

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
        # The model runs synchronously on CPU: the KV a store reads is written.
        job = _Job(job_id, is_store, device_ptrs, obj_names, time.perf_counter())
        with self._cond:
            if not obj_names or self._broken:
                self._results.append(TransferResult(job_id, not self._broken))
                return True
            self._jobs[job_id] = job
            if is_store:
                self._waiting_stores.append(job)
            elif expedited:
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
        """Wait until these stores have read the KV cache, and these loads
        have completed."""

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
        finally:
            self._client.shutdown()

    # Driver thread

    def _run(self) -> None:
        with self._cond:
            while not self._stop:
                try:
                    self._step()
                except Exception:
                    logger.exception("Object store driver failed")
                    self._fail_all()
                    return
                if self._pieces:
                    self._cond.wait(1e-4)
                elif not (self._waiting_loads or self._waiting_stores):
                    self._cond.wait()

    def _step(self) -> None:
        for obj_result in self._client.poll():
            piece_id = obj_result.job_id
            piece = self._pieces[piece_id]
            job = piece.job
            if not job.is_store:
                self._gets_in_flight -= len(piece.chunks)
            if not obj_result.success:
                if not job.failed:
                    logger.warning(
                        "Object store %s of job %d failed",
                        "store" if job.is_store else "load",
                        job.job_id,
                    )
                job.failed = True
            if not job.is_store and not job.failed:
                self._copy(piece, to_kv=True)
            self._finish_piece(piece_id)
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
            if not job.is_store:
                num_chunks = min(
                    num_chunks, self._max_inflight_gets - self._gets_in_flight
                )
                if num_chunks <= 0:
                    return
            slots = self._free_slots[-num_chunks:]
            del self._free_slots[-num_chunks:]
            end = job.next_chunk + num_chunks
            piece = _Piece(job, list(range(job.next_chunk, end)), slots)
            job.next_chunk = end
            job.in_flight += 1
            piece_id = self._next_piece_id
            self._next_piece_id += 1
            self._pieces[piece_id] = piece
            op = NIXL_WRITE
            if job.is_store:
                job.copying += 1
                self._copy(piece, to_kv=False)
                job.copying -= 1
                self._cond.notify_all()
            else:
                op = NIXL_READ
                self._gets_in_flight += num_chunks
            self._client.submit(
                piece_id, np.asarray(slots, dtype=np.int32), piece.obj_names, op
            )

    def _copy(self, piece: _Piece, to_kv: bool) -> None:
        """Copy a piece between the KV cache and its slots, without the lock:
        the engine thread keeps submitting meanwhile."""
        device_ptrs = select_device_chunks(
            piece.job.device_ptrs, self._blocks_per_chunk, piece.chunks
        )
        kv_ptrs, staging_ptrs, sizes = self._ops.build(device_ptrs, piece.slots)
        self._cond.release()
        try:
            if to_kv:
                _memmove_all(kv_ptrs, staging_ptrs, sizes)
            else:
                _memmove_all(staging_ptrs, kv_ptrs, sizes)
        finally:
            self._cond.acquire()

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
                    transfer_size=job.num_chunks * self._chunk_bytes,
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
