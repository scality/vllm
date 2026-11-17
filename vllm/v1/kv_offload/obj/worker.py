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
from collections.abc import Callable
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
    _build_copy_plan,
    _canonical_block_sizes,
    _select_swap_blocks_fn,
)
from vllm.v1.kv_offload.obj.client import NIXL_READ, NIXL_WRITE, ObjStoreClient
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec, owner_of
from vllm.v1.kv_offload.obj.peers import AllGather, CrossRankFence, share_tensors
from vllm.v1.kv_offload.pointer_utils import (
    remap_device_pointers,
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


def _keep_all(ref: CanonicalKVCacheRef) -> bool:
    return True


def _first_writer(ref: CanonicalKVCacheRef) -> bool:
    """Whether a store takes these bytes from this worker: one of the workers
    holding them is enough."""
    return ref.mapping is None or ref.mapping.writer_index == 0


def _copy_handler(
    ranks: list[int],
    staging_per_rank: list[list[torch.Tensor]],
    refs_per_rank: list[list[list[CanonicalKVCacheRef]]],
    keep: Callable[[CanonicalKVCacheRef], bool],
    blocks_per_chunk: int,
    gpu_to_cpu: bool,
    canonical_layout: bool,
    prefer_triton: bool,
) -> tuple[SingleDirectionOffloadingHandler, list[list[tuple[int, int]]]]:
    """One handler copying between the staging buffer and the KV caches of
    several workers, which use the same block ids, through the refs kept.

    Returns the handler and, per group, for each of its data refs the rank
    whose KV cache it is in and the index of the data ref in that rank's
    group, so that a transfer's pointers can be remapped to it."""
    num_tensors = 0
    staging_tensors: list[torch.Tensor] = []
    refs: list[list[CanonicalKVCacheRef]] = [[] for _ in refs_per_rank[0]]
    sources: list[list[tuple[int, int]]] = [[] for _ in refs_per_rank[0]]
    for rank, staging, rank_refs in zip(ranks, staging_per_rank, refs_per_rank):
        # Workers hold the same tensors: a data ref is in the same tensor in
        # every worker's KV cache.
        assert [[r.tensor_idx for r in g] for g in rank_refs] == [
            [r.tensor_idx for r in g] for g in refs_per_rank[0]
        ]
        kept = [
            [(d, ref) for d, ref in enumerate(group) if keep(ref)]
            for group in rank_refs
        ]
        used = sorted({ref.tensor_idx for group in kept for _, ref in group})
        index = {t: num_tensors + i for i, t in enumerate(used)}
        num_tensors += len(used)
        staging_tensors += [staging[t] for t in used]
        for group_refs, group_sources, group in zip(refs, sources, kept):
            group_refs += [
                dataclasses.replace(_sole_writer(ref), tensor_idx=index[ref.tensor_idx])
                for _, ref in group
            ]
            group_sources += [(rank, d) for d, _ in group]
    handler = SingleDirectionOffloadingHandler(
        cpu_tensors=staging_tensors,
        blocks_per_chunk=blocks_per_chunk,
        layer_refs_per_group=refs,
        gpu_to_cpu=gpu_to_cpu,
        canonical_layout=canonical_layout,
        prefer_triton=prefer_triton,
    )
    return handler, sources


class _FastLoadCopies:
    """Copies loaded chunks from staging slots into several workers' KV
    caches, for chunks of one GPU block in one KV cache group.

    The copy handler builds a piece's descriptors layer by layer in Python,
    which costs milliseconds per piece with many layers and workers. Here the
    fragments of every layer are laid out once per worker, so a piece's
    descriptors are one numpy broadcast; each worker's copy then runs on its
    own stream.
    """

    def __init__(
        self,
        tensors_per_rank: list[list[torch.Tensor]],
        staging_offsets_per_rank: list[list[int]],
        refs_per_rank: list[list[CanonicalKVCacheRef]],
        canonical_layout: bool,
        staging: torch.Tensor,
        chunk_bytes: int,
        prefer_triton: bool,
    ):
        self._staging_ptr = staging.data_ptr()
        self._chunk_bytes = chunk_bytes
        # Per worker: offset in a slot, KV base pointer, KV row stride and
        # size of each fragment.
        self._targets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        self._targets = []
        for tensors, offsets, refs in zip(
            tensors_per_rank, staging_offsets_per_rank, refs_per_rank
        ):
            src_off: list[int] = []
            dst_base: list[int] = []
            stride: list[int] = []
            size: list[int] = []
            for ref in refs:
                tensor = tensors[ref.tensor_idx]
                slot_offset = offsets[ref.tensor_idx]
                if canonical_layout:
                    plan = _build_copy_plan(ref, gpu_to_cpu=False)
                    src_off += (slot_offset + plan.frag_offsets_src).tolist()
                    dst_base += (tensor.data_ptr() + plan.frag_offsets_dst).tolist()
                    size += plan.frag_sizes.tolist()
                    stride += [tensor.stride(0)] * plan.num_frags
                else:
                    src_off.append(slot_offset)
                    dst_base.append(tensor.data_ptr())
                    size.append(ref.page_size_bytes)
                    stride.append(tensor.stride(0))
            self._targets.append(
                (
                    np.asarray(src_off, dtype=np.uint64),
                    np.asarray(dst_base, dtype=np.uint64),
                    np.asarray(stride, dtype=np.uint64),
                    np.asarray(size, dtype=np.int64),
                )
            )
        self._copy = _select_swap_blocks_fn(
            [refs_per_rank[0]], gpu_to_cpu=False, prefer_triton=prefer_triton
        )
        self._streams = [
            current_platform.Stream(device=staging.device) for _ in tensors_per_rank
        ]

    def launch(
        self, slots: list[int], block_ids: np.ndarray, after: torch.cuda.Stream
    ) -> list[torch.Event]:
        """Copy the chunks in slots into the blocks block_ids of every
        worker, once after's work so far is done. Returns one event per
        worker, recorded when its copy is."""
        slot_ptrs = self._staging_ptr + np.asarray(slots, dtype=np.uint64) * np.uint64(
            self._chunk_bytes
        )
        blocks = block_ids.astype(np.uint64)
        events: list[torch.Event] = []
        for (src_off, dst_base, stride, size), stream in zip(
            self._targets, self._streams
        ):
            src = (slot_ptrs[:, None] + src_off[None, :]).ravel().view(np.int64)
            dst = (dst_base[None, :] + blocks[:, None] * stride[None, :]).ravel()
            sizes = np.tile(size, len(slots))
            stream.wait_stream(after)
            with current_platform.stream(stream):
                self._copy(
                    torch.from_numpy(src),
                    torch.from_numpy(dst.view(np.int64)),
                    torch.from_numpy(sizes),
                    is_src_access_order_any=True,
                )
                event = torch.Event()
                event.record(stream)
            events.append(event)
        return events

    def synchronize(self) -> None:
        for stream in self._streams:
            stream.synchronize()


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
    # Load: copies into the workers' KV caches not completed yet.
    copies_left: int = 0
    # Load, fast path: one event per worker written.
    events: list[torch.Event] = dataclasses.field(default_factory=list)

    @property
    def obj_names(self) -> list[str]:
        return [self.job.obj_names[c] for c in self.chunks]


class ObjOffloadingWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        chunk_bytes: int,
        rank_page_bytes: int,
        canonical_layout: bool,
        store_config: dict,
        io_threads: int,
        staging_bytes: int,
        agent_name: str,
        rank: int = 0,
        world_size: int = 1,
        all_gather: AllGather | None = None,
        replicated: bool = False,
        piece_chunks: int | None = None,
        max_inflight_gets: int | None = None,
        prefer_triton: bool = False,
    ):
        """Args:
        kv_caches: Canonicalized KV caches.
        blocks_per_chunk: GPU blocks per chunk.
        chunk_bytes: Bytes of one object: a CPU tier chunk row, padding
            included.
        rank_page_bytes: Bytes of a worker's pages in a chunk, worker r's
            starting at r * rank_page_bytes as in the CPU tier (direct
            layout); 0 for the canonical or replicated layouts.
        canonical_layout: Chunks use the canonical layout.
        store_config: Object store parameters (see ObjStoreConfig).
        io_threads: NIXL I/O threads.
        staging_bytes: Size of the staging buffer.
        agent_name: NIXL agent name.
        rank: This worker's rank.
        world_size: Workers whose KV makes up a chunk, or hold it whole when
            replicated.
        all_gather: Exchanges objects between the workers, required with
            more than one.
        replicated: Every worker holds the whole chunk.
        piece_chunks: Most chunks a piece moves through the staging buffer
            (default a quarter of its slots).
        max_inflight_gets: Most chunks being read from the store at once
            (default all the slots): more only queue in the store.
        prefer_triton: Copy loaded chunks into the KV caches with the Triton
            kernel whatever the page size (see cpu_to_gpu_copy).

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
        self._max_piece_chunks = piece_chunks or max(1, num_slots // 4)
        self._max_inflight_gets = max_inflight_gets or num_slots
        self._device = kv_caches.tensors[0].tensor.device

        self._staging = torch.empty(
            (num_slots, chunk_bytes), dtype=torch.int8, device=self._device
        )
        gpu_tensors = [
            t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
            for t in kv_caches.tensors
        ]
        staging_page_sizes = self._staging_page_sizes(
            kv_caches, blocks_per_chunk, canonical_layout
        )
        self._fence: CrossRankFence | None = None
        # Stores read other workers' KV caches.
        self._sharded = world_size > 1 and not replicated
        if world_size == 1:
            all_tensors = [gpu_tensors]
            all_refs = [kv_caches.group_data_refs]
        else:
            assert all_gather is not None
            all_tensors = share_tensors(gpu_tensors, rank, all_gather)
            all_refs = all_gather(kv_caches.group_data_refs)
            self._fence = CrossRankFence(rank, world_size, all_gather)
        all_staging = [
            self._staging_views(staging_page_sizes, r * rank_page_bytes)
            for r in range(world_size)
        ]

        def copies(
            ranks: list[int], gpu_to_cpu: bool
        ) -> tuple[SingleDirectionOffloadingHandler, list[list[tuple[int, int]]]]:
            # A sharded store needs one copy of canonical bytes several
            # workers hold; a load writes them in every worker.
            if gpu_to_cpu and self._sharded and canonical_layout:
                keep = _first_writer
            else:
                keep = _keep_all
            return _copy_handler(
                ranks,
                [all_staging[r] for r in ranks],
                [all_refs[r] for r in ranks],
                keep,
                blocks_per_chunk,
                gpu_to_cpu,
                canonical_layout,
                prefer_triton=prefer_triton and not gpu_to_cpu,
            )

        # A replicated chunk is stored from the owner's own copy.
        store_ranks = list(range(world_size)) if self._sharded else [rank]
        self._store_copies, store_sources = copies(store_ranks, gpu_to_cpu=True)
        # The connector resolves a transfer's pointers in this worker's KV
        # cache; the handlers address the KV caches of the ranks they copy.
        self._ref_tensors = [
            [ref.tensor_idx for ref in group] for group in kv_caches.group_data_refs
        ]
        own_sources = [
            [(rank, d) for d in range(len(group))] for group in self._ref_tensors
        ]
        self._store_sources = None if store_sources == own_sources else store_sources
        self._own_bases = [t.data_ptr() for t in gpu_tensors]
        self._own_strides = [t.tensor.stride(0) for t in kv_caches.tensors]
        self._bases = [[t.data_ptr() for t in tensors] for tensors in all_tensors]
        self._fast_loads: _FastLoadCopies | None = None
        self._load_copies: list[
            tuple[SingleDirectionOffloadingHandler, list[list[tuple[int, int]]] | None]
        ] = []
        if blocks_per_chunk == 1 and len(kv_caches.group_data_refs) == 1:
            page_offsets = np.cumsum([0] + staging_page_sizes[:-1]).tolist()
            self._fast_loads = _FastLoadCopies(
                all_tensors,
                [
                    [r * rank_page_bytes + offset for offset in page_offsets]
                    for r in range(world_size)
                ],
                [refs[0] for refs in all_refs],
                canonical_layout,
                self._staging,
                chunk_bytes,
                prefer_triton,
            )
        else:
            # One handler, so one stream, per worker written: a load's copies
            # to the workers run at once.
            for r in range(world_size):
                handler, sources = copies([r], gpu_to_cpu=False)
                self._load_copies.append(
                    (handler, None if sources == own_sources else sources)
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
        # Sharded: store jobs in submission order, None for those this worker
        # owns nothing of, until they have read the KV caches.
        self._uncopied_stores: deque[tuple[int, _Job | None]] = deque()
        self._pieces: dict[int, _Piece] = {}
        # Load pieces whose fast-path copies are in flight.
        self._fast_copying: dict[int, _Piece] = {}
        self._next_piece_id = 0
        # Chunks of load pieces being read from the store.
        self._gets_in_flight = 0
        self._results: list[TransferResult] = []
        self._stop = False
        # Set when the driver died: every job fails from then on.
        self._broken = False

        self._stream = current_platform.Stream(device=self._device)
        self._thread = threading.Thread(
            target=self._run, name="vllm_offloading_obj_driver", daemon=True
        )
        self._thread.start()

    @staticmethod
    def _staging_page_sizes(
        kv_caches: CanonicalKVCaches, blocks_per_chunk: int, canonical_layout: bool
    ) -> list[int]:
        if canonical_layout:
            sizes = _canonical_block_sizes(
                kv_caches.group_data_refs, len(kv_caches.tensors)
            )
        else:
            sizes = [t.page_size_bytes for t in kv_caches.tensors]
        return [size * blocks_per_chunk for size in sizes]

    def _staging_views(self, page_sizes: list[int], offset: int) -> list[torch.Tensor]:
        """Views of the staging buffer laid out like a CPU tier chunk row (see
        SharedOffloadRegion), a worker's pages starting at offset."""
        views: list[torch.Tensor] = []
        for page_size in page_sizes:
            assert offset + page_size <= self._chunk_bytes
            views.append(
                self._staging.as_strided(
                    (self._staging.shape[0], page_size),
                    (self._chunk_bytes, 1),
                    storage_offset=offset,
                )
            )
            offset += page_size
        return views

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
        if self._fence is not None and (self._sharded or not is_store):
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
            if is_store and self._sharded:
                self._uncopied_stores.append((job_id, job if chunks else None))
                self._publish_copied()
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

    def _publish_copied(self) -> None:
        """Publish how far this worker's stores have read the KV caches."""
        assert self._fence is not None
        copied = None
        while self._uncopied_stores:
            job_id, job = self._uncopied_stores[0]
            if job is not None and not (job.started and job.copying == 0):
                break
            self._uncopied_stores.popleft()
            copied = job_id
        if copied is not None:
            self._fence.publish_copied(copied)

    def get_finished(self) -> list[TransferResult]:
        with self._cond:
            results, self._results = self._results, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Wait until these stores have read the KV caches, and these loads
        have completed. Sharded, the stores other workers own included."""

        def settled(job: _Job | None) -> bool:
            if job is None:
                return True
            return job.is_store and job.started and job.copying == 0

        with self._cond:
            self._cond.wait_for(
                lambda: all(settled(self._jobs.get(j)) for j in job_ids)
            )
        if self._sharded and job_ids:
            assert self._fence is not None
            self._fence.wait_copied(max(job_ids), _FENCE_TIMEOUT_S)

    def shutdown(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join()
        try:
            self._client.drain()
            self._stream.synchronize()
            self._store_copies.shutdown()
            for handler, _ in self._load_copies:
                handler.shutdown()
            if self._fast_loads is not None:
                self._fast_loads.synchronize()
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
        for piece_id, piece in list(self._fast_copying.items()):
            if all(event.query() for event in piece.events):
                del self._fast_copying[piece_id]
                self._finish_piece(piece_id)
        for handler, _ in self._load_copies:
            for result in handler.get_finished():
                piece = self._pieces[result.job_id]
                if not result.success and not piece.job.failed:
                    logger.warning(
                        "Object store job %d failed: copy into the KV cache",
                        piece.job.job_id,
                    )
                    piece.job.failed = True
                piece.copies_left -= 1
                if piece.copies_left == 0:
                    self._finish_piece(result.job_id)
        for obj_result in self._client.poll():
            piece = self._pieces[obj_result.job_id]
            if not piece.job.is_store:
                self._gets_in_flight -= len(piece.chunks)
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
        if self._sharded:
            self._publish_copied()

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
            piece = _Piece(job, job.chunks[job.next_chunk : end], slots)
            job.next_chunk = end
            job.in_flight += 1
            piece_id = self._next_piece_id
            self._next_piece_id += 1
            self._pieces[piece_id] = piece
            if job.is_store:
                self._copy(piece_id, piece)
            else:
                self._gets_in_flight += num_chunks
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
            # Workers use the same block ids: the handlers copy them in every
            # worker's tensors, once each has received the job.
            if self._fence is not None and (self._sharded or not job.is_store):
                self._fence.wait(job.job_id, self._stream, _FENCE_TIMEOUT_S)
            if job.is_store:
                job.copying += 1
                self._store_copies.transfer_async(
                    piece_id, self._remap(device_ptrs, self._store_sources), slots
                )
            elif self._fast_loads is not None:
                piece.events = self._fast_loads.launch(
                    piece.slots, self._block_ids(device_ptrs), self._stream
                )
                self._fast_copying[piece_id] = piece
            else:
                piece.copies_left = len(self._load_copies)
                for handler, sources in self._load_copies:
                    handler.transfer_async(
                        piece_id, self._remap(device_ptrs, sources), slots
                    )

    def _remap(
        self, device_ptrs: DevicePointers, sources: list[list[tuple[int, int]]] | None
    ) -> DevicePointers:
        """Pointers into the KV caches a handler copies, from this worker's."""
        if sources is None:
            return device_ptrs
        return remap_device_pointers(
            device_ptrs, self._ref_tensors, sources, self._own_bases, self._bases
        )

    def _block_ids(self, device_ptrs: DevicePointers) -> np.ndarray:
        """The GPU blocks of a transfer of one KV cache group, from the
        pointers of its first data ref, as resolve_device_pointers() made
        them."""
        assert len(device_ptrs.group_block_counts) == 1
        num_blocks = device_ptrs.group_block_counts[0]
        t_idx = self._ref_tensors[0][0]
        offsets = device_ptrs.ptrs[:num_blocks] - np.uint64(self._own_bases[t_idx])
        return offsets // np.uint64(self._own_strides[t_idx])

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
        self._fast_copying.clear()
        self._cond.notify_all()
