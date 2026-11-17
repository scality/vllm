# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of offloading KV straight from the GPU to an object store.

A store copies the chunks' KV into slots of a staging buffer in GPU memory, in
the CPU tier's chunk layout, then PUTs each slot as one object over GPUDirect
RDMA. A load GETs the objects into slots, then copies them into the KV cache.
A job larger than the staging buffer goes through it in pieces. A driver
thread moves pieces from one stage to the next as soon as each completes.
"""

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
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.pointer_utils import slice_device_pointers

logger = init_logger(__name__)

# How long the driver waits between polls while transfers are in flight.
_POLL_INTERVAL_S = 1e-4


class StagingSlots(BlockIDsLoadStoreSpec):
    """Slots of the staging buffer, one per chunk."""


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
    next_chunk: int = 0
    in_flight: int = 0
    # Store: pieces being copied out of the KV cache.
    copying: int = 0
    failed: bool = False

    @property
    def num_chunks(self) -> int:
        return len(self.obj_names)

    @property
    def started(self) -> bool:
        return self.failed or self.next_chunk == self.num_chunks


@dataclass(eq=False)
class _Piece:
    job: _Job
    start: int
    end: int
    slots: list[int]


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

        """
        self._blocks_per_chunk = blocks_per_chunk
        self._chunk_bytes = chunk_bytes
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
        self._store_copies = SingleDirectionOffloadingHandler(
            cpu_tensors=staging_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )
        self._load_copies = SingleDirectionOffloadingHandler(
            cpu_tensors=staging_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
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
        job = _Job(job_id, is_store, device_ptrs, obj_names, ready, time.perf_counter())
        with self._cond:
            if not obj_names or self._broken:
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
                piece.job.obj_names[piece.start : piece.end],
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
            piece = _Piece(job, job.next_chunk, job.next_chunk + num_chunks, slots)
            job.next_chunk = piece.end
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
                    job.obj_names[piece.start : piece.end],
                    NIXL_READ,
                )

    def _copy(self, piece_id: int, piece: _Piece) -> None:
        """Copy a piece between the KV cache and its slots."""
        job = piece.job
        device_ptrs = slice_device_pointers(
            job.device_ptrs, self._blocks_per_chunk, piece.start, piece.end
        )
        slots = StagingSlots(piece.slots)
        with current_platform.stream(self._stream):
            self._stream.wait_event(job.ready)
            if job.is_store:
                job.copying += 1
                self._store_copies.transfer_async(piece_id, device_ptrs, slots)
            else:
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
        self._cond.notify_all()
