# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of offloading KV straight from a CPU KV cache to an object store.

The CPU counterpart of ObjOffloadingWorker, for one worker and the direct
layout. A store copies the chunks' KV into slots of a staging buffer in host
memory, in the CPU tier's chunk layout, then PUTs each slot as one object over
RDMA. A load GETs the objects into slots, then copies them into the KV cache.
A job larger than the staging buffer goes through it in pieces. A driver
thread moves pieces from one stage to the next; a pool of threads runs the
copies, which release the GIL, sharing each piece between them. The offload
threads run on the CPUs vLLM's OpenMP threads are not bound to.
"""

import ctypes
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import regex as re
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
    # Seconds the pieces spent in transfers and in copies, summed.
    transfer_s: float = 0.0
    copy_s: float = 0.0

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
    # When its current stage, transfer or copy, started.
    stage_start: float = 0.0

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


# Longest range one copy thread moves at once, so a piece's bytes can be
# shared evenly between the threads.
_MAX_RUN_BYTES = 4 * 2**20


def _runs(
    dst: np.ndarray, src: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ranges to copy, merged where contiguous on both sides, then cut to
    at most _MAX_RUN_BYTES."""
    breaks = np.flatnonzero(
        (dst[1:] != dst[:-1] + sizes[:-1]) | (src[1:] != src[:-1] + sizes[:-1])
    )
    starts = np.concatenate(([0], breaks + 1))
    run_bytes = np.add.reduceat(sizes, starts).astype(np.int64)
    pieces = (run_bytes + _MAX_RUN_BYTES - 1) // _MAX_RUN_BYTES
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    offsets = (np.arange(int(pieces.sum())) - first) * _MAX_RUN_BYTES
    lengths = np.minimum(np.repeat(run_bytes, pieces) - offsets, _MAX_RUN_BYTES)
    offsets = offsets.astype(np.uint64)
    return (
        np.repeat(dst[starts], pieces) + offsets,
        np.repeat(src[starts], pieces) + offsets,
        lengths,
    )


def _memmove_runs(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray) -> None:
    for d, s, n in zip(dst.tolist(), src.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)


def _parse_cpu_list(text: str) -> set[int]:
    """CPUs of a list like "0-3,8" or "0 1 2"."""
    cpus: set[int] = set()
    for item in re.split(r"[,\s]+", text.strip()):
        if "-" in item:
            first, last = item.split("-")
            cpus.update(range(int(first), int(last) + 1))
        elif item:
            cpus.add(int(item))
    return cpus


def _offload_cpus() -> set[int] | None:
    """CPUs for the offload threads: the container's that vLLM's OpenMP threads
    are not bound to (all of them when that leaves none), or None where
    affinity cannot be set. A thread inherits the affinity of the thread
    creating it, which OpenMP binding narrows to one core."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    allowed: set[int] = set()
    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
    ):
        try:
            with open(path) as f:
                allowed = _parse_cpu_list(f.read())
            break
        except (OSError, ValueError):
            continue
    if not allowed:
        allowed = set(range(os.cpu_count() or 1))
    omp: set[int] = set()
    try:
        kmp = re.search(r"proclist=\[([^\]]*)\]", os.environ.get("KMP_AFFINITY", ""))
        if kmp:
            omp = _parse_cpu_list(kmp.group(1))
        elif "GOMP_CPU_AFFINITY" in os.environ:
            omp = _parse_cpu_list(os.environ["GOMP_CPU_AFFINITY"])
        elif "OMP_PLACES" in os.environ:
            omp = _parse_cpu_list(os.environ["OMP_PLACES"].strip("{}"))
    except ValueError:
        omp = set()
    return (allowed - omp) or allowed


def _pin(cpus: set[int] | None) -> None:
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            logger.debug("Cannot pin an object store offload thread to %s", cpus)


class _Copier:
    """Copies ranges with a pool of threads, sharing each copy between them."""

    def __init__(self, threads: int, cpus: set[int] | None):
        self._threads = threads
        self._pool = ThreadPoolExecutor(
            threads,
            thread_name_prefix="vllm_offloading_obj_copy",
            initializer=_pin,
            initargs=(cpus,),
        )

    def submit(
        self, dst: np.ndarray, src: np.ndarray, sizes: np.ndarray
    ) -> list[Future]:
        if len(sizes) == 0:
            return []
        dst, src, sizes = _runs(dst, src, sizes)
        # Contiguous groups of runs of about equal bytes, one per thread.
        ends = np.cumsum(sizes)
        bounds = np.searchsorted(
            ends, ends[-1] * np.arange(1, self._threads) / self._threads
        )
        edges = [0, *np.unique(bounds).tolist(), len(sizes)]
        return [
            self._pool.submit(_memmove_runs, dst[a:b], src[a:b], sizes[a:b])
            for a, b in zip(edges, edges[1:])
            if b > a
        ]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


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
        copy_threads: int = 8,
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
        copy_threads: Threads copying between the KV cache and the staging
            buffer.

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
        self._cpus = _offload_cpus()
        self._copier = _Copier(copy_threads, self._cpus)

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
            "Object store staging: %d slots of %d bytes in host memory, "
            "%d copy threads on %s CPUs",
            num_slots,
            chunk_bytes,
            copy_threads,
            len(self._cpus) if self._cpus else "inherited",
        )

        # Everything below is guarded by _cond.
        self._cond = threading.Condition()
        self._free_slots: list[int] = list(range(num_slots))
        self._jobs: dict[int, _Job] = {}
        self._waiting_loads: deque[_Job] = deque()
        self._waiting_stores: deque[_Job] = deque()
        self._pieces: dict[int, _Piece] = {}
        # Pieces being copied: their copy threads' futures.
        self._copying: dict[int, list[Future]] = {}
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
            self._copier.shutdown()
            self._client.drain()
        finally:
            self._client.shutdown()

    # Driver thread

    def _run(self) -> None:
        _pin(self._cpus)
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
        now = time.perf_counter()
        for obj_result in self._client.poll():
            piece_id = obj_result.job_id
            piece = self._pieces[piece_id]
            job = piece.job
            job.transfer_s += now - piece.stage_start
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
                self._start_copy(piece_id, piece, to_kv=True)
            else:
                self._finish_piece(piece_id)
        for piece_id, futures in list(self._copying.items()):
            if all(f.done() for f in futures):
                del self._copying[piece_id]
                self._copied(piece_id, futures)
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
            if job.is_store:
                job.copying += 1
                self._start_copy(piece_id, piece, to_kv=False)
            else:
                self._gets_in_flight += num_chunks
                self._transfer(piece_id, piece, NIXL_READ)

    def _transfer(self, piece_id: int, piece: _Piece, op: str) -> None:
        piece.stage_start = time.perf_counter()
        self._client.submit(
            piece_id, np.asarray(piece.slots, dtype=np.int32), piece.obj_names, op
        )

    def _start_copy(self, piece_id: int, piece: _Piece, to_kv: bool) -> None:
        """Have the copy threads move a piece between the KV cache and its
        slots; the driver keeps going meanwhile."""
        device_ptrs = select_device_chunks(
            piece.job.device_ptrs, self._blocks_per_chunk, piece.chunks
        )
        kv_ptrs, staging_ptrs, sizes = self._ops.build(device_ptrs, piece.slots)
        piece.stage_start = time.perf_counter()
        if to_kv:
            futures = self._copier.submit(kv_ptrs, staging_ptrs, sizes)
        else:
            futures = self._copier.submit(staging_ptrs, kv_ptrs, sizes)
        self._copying[piece_id] = futures

    def _copied(self, piece_id: int, futures: list[Future]) -> None:
        piece = self._pieces[piece_id]
        job = piece.job
        job.copy_s += time.perf_counter() - piece.stage_start
        error = next((f.exception() for f in futures if f.exception()), None)
        if error is not None and not job.failed:
            logger.warning(
                "Object store job %d failed: copy %s the KV cache: %s",
                job.job_id,
                "out of" if job.is_store else "into",
                error,
            )
        job.failed = job.failed or error is not None
        if job.is_store:
            job.copying -= 1
            self._cond.notify_all()
            if not job.failed:
                self._transfer(piece_id, piece, NIXL_WRITE)
                return
        self._finish_piece(piece_id)

    def _finish_piece(self, piece_id: int) -> None:
        piece = self._pieces.pop(piece_id)
        self._free_slots.extend(piece.slots)
        job = piece.job
        job.in_flight -= 1
        if job.in_flight == 0 and job.started:
            del self._jobs[job.job_id]
            elapsed = time.perf_counter() - job.submit_time
            size = job.num_chunks * self._chunk_bytes
            logger.debug(
                "Object store %s job %d: %d chunks, %.1f MB in %.1f ms "
                "(%.2f GB/s); transfers %.1f ms, copies %.1f ms, summed over "
                "pieces",
                "store" if job.is_store else "load",
                job.job_id,
                job.num_chunks,
                size / 1e6,
                elapsed * 1e3,
                size / elapsed / 1e9,
                job.transfer_s * 1e3,
                job.copy_s * 1e3,
            )
            self._results.append(
                TransferResult(
                    job_id=job.job_id,
                    success=not job.failed,
                    transfer_size=size,
                    transfer_time=elapsed,
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
        self._copying.clear()
        self._cond.notify_all()
