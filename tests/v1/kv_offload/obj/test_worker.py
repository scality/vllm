# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    CanonicalPageMapping,
    CopyRun,
    DevicePointers,
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.obj.client import NIXL_WRITE, ObjTransferResult
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec, owner_of
from vllm.v1.kv_offload.obj.cpu_worker import ObjCpuOffloadingWorker
from vllm.v1.kv_offload.obj.worker import ObjOffloadingWorker
from vllm.v1.kv_offload.pointer_utils import (
    remap_device_pointers,
    resolve_device_pointers,
    select_device_chunks,
)


def _pointers(
    ptrs: list[int], group_sizes: list[int], refs: list[int], block_indices: list[int]
) -> DevicePointers:
    return DevicePointers(
        ptrs=np.array(ptrs, dtype=np.uint64),
        sizes=np.full(len(ptrs), 8, dtype=np.uint64),
        group_block_counts=tuple(group_sizes),
        group_data_ref_counts=tuple(refs),
        block_indices=tuple(block_indices),
    )


def _blocks(ptrs: DevicePointers) -> tuple[list[int], list[int], list[int]]:
    return (
        ptrs.ptrs.tolist(),
        list(ptrs.group_block_counts),
        list(ptrs.block_indices),
    )


@pytest.mark.cpu_test
def test_slice_starts_mid_chunk():
    """The first chunk holds only the blocks past the request's offset in it;
    a slice after it starts on a chunk boundary."""
    ptrs = _pointers([10, 11, 12, 13, 14], [5], [1], [1])

    assert _blocks(select_device_chunks(ptrs, 2, [0])) == ([10], [1], [1])
    assert _blocks(select_device_chunks(ptrs, 2, [1, 2])) == (
        [11, 12, 13, 14],
        [4],
        [2],
    )


@pytest.mark.cpu_test
def test_slice_numbers_chunks_group_by_group():
    ptrs = _pointers([10, 11, 20, 21, 22], [2, 3], [1, 1], [0, 0])

    assert _blocks(select_device_chunks(ptrs, 1, [1, 2])) == ([11, 20], [1, 1], [1, 0])
    assert _blocks(select_device_chunks(ptrs, 1, [3, 4])) == ([21, 22], [0, 2], [0, 1])


@pytest.mark.cpu_test
def test_select_keeps_every_data_ref_of_a_block():
    """Pointers are laid out ref by ref within a group: a selection takes the
    same blocks from each ref."""
    ptrs = _pointers([10, 11, 12, 20, 21, 22], [3], [2], [0])

    assert _blocks(select_device_chunks(ptrs, 1, [1, 2])) == (
        [11, 12, 21, 22],
        [2],
        [1],
    )


@pytest.mark.cpu_test
def test_select_skips_chunks_owned_by_other_workers():
    """The chunks a worker owns need not be consecutive; their blocks fill
    its slots in order."""
    ptrs = _pointers([10, 11, 12, 13, 14], [5], [1], [1])

    assert _blocks(select_device_chunks(ptrs, 2, [0, 2])) == ([10, 13, 14], [3], [1])


def _two_ref_pointers() -> DevicePointers:
    """This worker's pointers: its tensors are at 1000 and 2000. Group 0 has
    blocks 1 and 2 of 8 bytes in both tensors, group 1 none."""
    return DevicePointers(
        ptrs=np.array([1008, 1016, 2008, 2016], dtype=np.uint64),
        sizes=np.full(4, 8, dtype=np.uint64),
        group_block_counts=(2, 0),
        group_data_ref_counts=(2, 1),
        block_indices=(0, 0),
    )


@pytest.mark.cpu_test
def test_remapped_pointers_address_every_worker():
    """A replicated load writes the same blocks into each worker's tensors,
    worker by worker as the handler's refs are ordered."""
    remapped = remap_device_pointers(
        _two_ref_pointers(),
        data_ref_tensors=[[0, 1], [1]],
        sources=[[(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (1, 0)]],
        own_bases=[1000, 2000],
        bases=[[5000, 6000], [1000, 2000]],
    )

    assert remapped.ptrs.tolist() == [
        5008, 5016, 6008, 6016,  # rank 0's KV cache
        1008, 1016, 2008, 2016,  # rank 1's: this worker's
    ]  # fmt: skip
    assert remapped.group_data_ref_counts == (4, 2)
    assert remapped.group_block_counts == (2, 0)


@pytest.mark.cpu_test
def test_remapped_pointers_keep_only_the_refs_kept():
    """A sharded store reads bytes several workers hold from one of them."""
    remapped = remap_device_pointers(
        _two_ref_pointers(),
        data_ref_tensors=[[0, 1], [1]],
        sources=[[(0, 1), (1, 0)], []],
        own_bases=[1000, 2000],
        bases=[[5000, 6000], [1000, 2000]],
    )

    assert remapped.ptrs.tolist() == [6008, 6016, 1008, 1016]
    assert remapped.group_data_ref_counts == (2, 0)


@pytest.mark.cpu_test
def test_fast_load_finds_the_blocks_of_its_pointers():
    """The fast load path takes block ids: they are found back from the
    pointers of the first data ref, as resolve_device_pointers() made them."""
    worker = SimpleNamespace(
        _ref_tensors=[[1, 0]], _own_bases=[5000, 1000], _own_strides=[64, 32]
    )
    ptrs = DevicePointers(
        ptrs=np.array(
            [1000 + 7 * 32, 1000, 1000 + 5 * 32, 5448, 5000, 5320], dtype=np.uint64
        ),
        sizes=np.full(6, 8, dtype=np.uint64),
        group_block_counts=(3,),
        group_data_ref_counts=(2,),
        block_indices=(0,),
    )

    block_ids = ObjOffloadingWorker._block_ids(worker, ptrs)  # type: ignore[arg-type]

    assert block_ids.tolist() == [7, 0, 5]


@pytest.mark.cpu_test
def test_every_chunk_has_one_owner_in_every_process():
    names = [f"model/{i:03x}/{i}.bin" for i in range(64)]
    owners = [owner_of(n, 8) for n in names]

    assert owners == [owner_of(n, 8) for n in names]
    assert set(owners) == set(range(8))


def _synchronize() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


class _FakeStore:
    """An object store holding objects in a dict, moving slots of the
    staging buffer synchronously."""

    def __init__(self, *args, **kwargs):
        self.objects: dict[str, torch.Tensor] = {}
        self.staging: torch.Tensor | None = None
        self._results: list[ObjTransferResult] = []

    def register_local(self, **kwargs) -> None:
        pass

    def submit(self, job_id, slot_ids, obj_names, op) -> None:
        assert self.staging is not None
        _synchronize()
        success = True
        for slot, name in zip(slot_ids.tolist(), obj_names):
            if op == NIXL_WRITE:
                self.objects[name] = self.staging[slot].clone()
            elif name in self.objects:
                self.staging[slot].copy_(self.objects[name])
            else:
                success = False
        _synchronize()
        self._results.append(ObjTransferResult(job_id, success))

    def poll(self) -> list[ObjTransferResult]:
        results, self._results = self._results, []
        return results

    def drain(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


PAGE = 256
BLOCKS_PER_CHUNK = 2
CHUNK_BYTES = 4096


@pytest.fixture
def kv() -> torch.Tensor:
    if not torch.accelerator.is_available():
        pytest.skip("needs a GPU")
    return torch.randint(-128, 127, (8, PAGE), dtype=torch.int8, device="cuda")


def _kv_caches(kv: torch.Tensor) -> CanonicalKVCaches:
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(kv, PAGE)],
        group_data_refs=[[CanonicalKVCacheRef(0, PAGE)]],
    )


@pytest.fixture
def worker(kv: torch.Tensor):
    kv_caches = _kv_caches(kv)
    with patch("vllm.v1.kv_offload.obj.worker.ObjStoreClient", _FakeStore):
        worker = ObjOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            chunk_bytes=CHUNK_BYTES,
            rank_page_bytes=0,
            canonical_layout=False,
            store_config={},
            io_threads=1,
            staging_bytes=CHUNK_BYTES,  # one slot: a job goes in pieces
            agent_name="test",
        )
    worker._client.staging = worker._staging  # type: ignore[attr-defined]
    yield worker
    worker.shutdown()


def _finish(worker: ObjOffloadingWorker, job_id: int) -> TransferResult:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for result in worker.get_finished():
            assert result.job_id == job_id
            return result
        time.sleep(0.001)
    raise TimeoutError(job_id)


def _ptrs(kv: torch.Tensor, block_ids: list[int]) -> DevicePointers:
    spec = GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])
    return resolve_device_pointers(spec, _kv_caches(kv))


def test_store_then_load_round_trips(worker, kv):
    """Each object is a CPU tier chunk row: the chunk's pages, then padding."""
    names = ObjLoadStoreSpec(["a", "b"])
    expected = kv[[2, 3, 4, 5]].clone()

    worker.submit_store(1, _ptrs(kv, [2, 3, 4, 5]), names)
    assert _finish(worker, 1).success
    objects = worker._client.objects
    assert torch.equal(objects["a"][: 2 * PAGE].view(2, PAGE), expected[:2])
    assert torch.equal(objects["b"][: 2 * PAGE].view(2, PAGE), expected[2:])

    kv.zero_()
    worker.submit_load(2, names, _ptrs(kv, [6, 7, 0, 1]))
    assert _finish(worker, 2).success
    assert torch.equal(kv[[6, 7, 0, 1]], expected)


def test_load_of_a_missing_object_fails(worker, kv):
    worker.submit_load(3, ObjLoadStoreSpec(["missing"]), _ptrs(kv, [0, 1]))

    assert not _finish(worker, 3).success


def test_wait_returns_once_the_kv_is_read(worker, kv):
    worker.submit_store(4, _ptrs(kv, [0, 1, 2, 3]), ObjLoadStoreSpec(["c", "d"]))
    worker.wait({4})

    assert set(worker._client.objects) <= {"c", "d"}
    assert _finish(worker, 4).success


def test_slots_are_shared_out_one_chunk_per_piece(worker):
    assert worker._max_piece_chunks == 1
    assert np.array_equal(worker._free_slots, [0])


def _replicated_worker(rank: int, port: int, result_queue) -> None:
    import torch.distributed as dist

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )

    def all_gather(obj):
        gathered = [None, None]
        dist.all_gather_object(gathered, obj)
        return gathered

    # Every worker holds the same KV, as with MLA.
    gen = torch.Generator().manual_seed(0)
    kv = torch.randint(-128, 127, (16, PAGE), dtype=torch.int8, generator=gen)
    kv = kv.cuda()
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(kv, PAGE)],
        group_data_refs=[[CanonicalKVCacheRef(0, PAGE)]],
    )
    with patch("vllm.v1.kv_offload.obj.worker.ObjStoreClient", _FakeStore):
        worker = ObjOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            chunk_bytes=CHUNK_BYTES,
            rank_page_bytes=0,
            canonical_layout=False,
            store_config={},
            io_threads=1,
            staging_bytes=2 * CHUNK_BYTES,
            agent_name=f"test{rank}",
            rank=rank,
            world_size=2,
            all_gather=all_gather,
            replicated=True,
        )
    worker._client.staging = worker._staging  # type: ignore[attr-defined]
    names = ObjLoadStoreSpec([f"chunk{i}" for i in range(4)])
    expected = kv[:8].clone()

    worker.submit_store(1, _ptrs(kv, list(range(8))), names)
    stored = _finish(worker, 1).success
    dist.barrier()
    kv[8:] = 0
    torch.accelerator.synchronize()
    dist.barrier()
    worker.submit_load(2, names, _ptrs(kv, list(range(8, 16))))
    loaded = _finish(worker, 2).success
    dist.barrier()  # every owner has written into every worker's KV cache
    result_queue.put(
        (
            rank,
            stored,
            loaded,
            set(worker._client.objects),  # type: ignore[attr-defined]
            torch.equal(kv[8:], expected),
        )
    )
    dist.barrier()
    worker.shutdown()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
def test_replicated_workers_share_out_the_chunks():
    """With identical KV on every worker, each chunk is stored by its owner
    alone, and a load writes every chunk into every worker's KV cache."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(_replicated_worker, args=(get_open_port(), result_queue), nprocs=2)
    results = sorted(result_queue.get() for _ in range(2))

    names = {f"chunk{i}" for i in range(4)}
    assert all(stored and loaded for _, stored, loaded, _, _ in results)
    assert results[0][3] | results[1][3] == names
    assert not results[0][3] & results[1][3]
    assert all(kv_equal for *_, kv_equal in results)


def _sharded_kv(rank: int) -> torch.Tensor:
    """Worker rank's half of the KV: different on every worker."""
    gen = torch.Generator().manual_seed(rank + 1)
    return torch.randint(-128, 127, (16, PAGE), dtype=torch.int8, generator=gen)


def _interleaving_mapping(rank: int) -> CanonicalPageMapping:
    """Each worker's page cut into 4 fragments, interleaved with the other
    worker's in a canonical page of both, as heads sharded by TP are."""
    fragment = PAGE // 4
    return CanonicalPageMapping(
        canonical_page_size_bytes=2 * PAGE,
        local_page_size_bytes=PAGE,
        runs=(
            CopyRun(
                local_offset=0,
                canonical_offset=rank * fragment,
                fragment_size=fragment,
                num_fragments=4,
                local_stride=fragment,
                canonical_stride=2 * fragment,
            ),
        ),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=True,
    )


def _expected_object(chunk: int, canonical: bool, bpc: int) -> torch.Tensor:
    """The object of chunk (blocks bpc * chunk to bpc * (chunk + 1))."""
    pages = [_sharded_kv(r)[bpc * chunk : bpc * (chunk + 1)] for r in range(2)]
    if canonical:
        fragments = [p.view(bpc, 4, PAGE // 4) for p in pages]
        body = torch.stack(fragments, dim=2).reshape(-1)
    else:
        body = torch.cat([p.reshape(-1) for p in pages])
    obj = torch.zeros(CHUNK_BYTES, dtype=torch.int8)
    obj[: body.numel()] = body
    return obj


def _sharded_worker(
    rank: int, port: int, canonical: bool, bpc: int, result_queue
) -> None:
    import torch.distributed as dist

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )

    def all_gather(obj):
        gathered = [None, None]
        dist.all_gather_object(gathered, obj)
        return gathered

    kv = _sharded_kv(rank).cuda()
    mapping = _interleaving_mapping(rank) if canonical else None
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(kv, PAGE)],
        group_data_refs=[[CanonicalKVCacheRef(0, PAGE, mapping)]],
    )
    with patch("vllm.v1.kv_offload.obj.worker.ObjStoreClient", _FakeStore):
        worker = ObjOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=bpc,
            chunk_bytes=CHUNK_BYTES,
            rank_page_bytes=0 if canonical else PAGE * bpc,
            canonical_layout=canonical,
            store_config={},
            io_threads=1,
            staging_bytes=2 * CHUNK_BYTES,
            agent_name=f"test{rank}",
            rank=rank,
            world_size=2,
            all_gather=all_gather,
        )
    worker._client.staging = worker._staging  # type: ignore[attr-defined]
    # One block per chunk takes the fast load path.
    assert (worker._fast_loads is not None) == (bpc == 1)
    names = ObjLoadStoreSpec([f"chunk{i}" for i in range(8 // bpc)])
    expected = kv[:8].clone()

    worker.submit_store(1, _ptrs(kv, list(range(8))), names)
    worker.wait({1})  # every owner has read this worker's blocks
    kv[:8] = 0  # so the store cannot have read them late
    stored = _finish(worker, 1).success
    dist.barrier()
    worker.submit_load(2, names, _ptrs(kv, list(range(8, 16))))
    loaded = _finish(worker, 2).success
    dist.barrier()  # every owner has written into every worker's KV cache
    # Bytes: a tensor would go through shared memory gone with this process.
    client = cast(_FakeStore, worker._client)
    objects = {n: o.cpu().numpy().tobytes() for n, o in client.objects.items()}
    result_queue.put((rank, stored, loaded, objects, torch.equal(kv[8:16], expected)))
    dist.barrier()
    worker.shutdown()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("bpc", [1, 2])
def test_sharded_chunks_are_gathered_and_scattered(canonical: bool, bpc: int):
    """With KV sharded across workers, a chunk's owner gathers every worker's
    part into one object, laid out as the CPU tier lays it out, and a load
    gives every worker its part back."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(
        _sharded_worker,
        args=(get_open_port(), canonical, bpc, result_queue),
        nprocs=2,
    )
    results = sorted(result_queue.get() for _ in range(2))

    assert all(stored and loaded for _, stored, loaded, _, _ in results)
    objects = {**results[0][3], **results[1][3]}
    assert sorted(objects) == sorted(f"chunk{i}" for i in range(8 // bpc))
    for i in range(8 // bpc):
        expected = _expected_object(i, canonical, bpc).numpy().tobytes()
        assert objects[f"chunk{i}"] == expected
    assert all(kv_equal for *_, kv_equal in results)


def test_one_block_chunks_take_the_fast_load_path(kv):
    """Chunks of one GPU block are loaded through the vectorized fast path,
    to the same bytes."""
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(kv, PAGE)],
        group_data_refs=[[CanonicalKVCacheRef(0, PAGE)]],
    )
    with patch("vllm.v1.kv_offload.obj.worker.ObjStoreClient", _FakeStore):
        worker = ObjOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            chunk_bytes=CHUNK_BYTES,
            rank_page_bytes=0,
            canonical_layout=False,
            store_config={},
            io_threads=1,
            staging_bytes=2 * CHUNK_BYTES,
            agent_name="test",
        )
    worker._client.staging = worker._staging  # type: ignore[attr-defined]
    try:
        assert worker._fast_loads is not None
        names = ObjLoadStoreSpec([f"c{i}" for i in range(3)])
        expected = kv[[1, 2, 3]].clone()
        worker.submit_store(1, _ptrs(kv, [1, 2, 3]), names)
        assert _finish(worker, 1).success
        kv.zero_()
        worker.submit_load(2, names, _ptrs(kv, [7, 0, 5]))
        assert _finish(worker, 2).success
        assert torch.equal(kv[[7, 0, 5]], expected)
    finally:
        worker.shutdown()


# CPU worker: the KV cache in host memory, two tensors of different page sizes.
CPU_PAGES = (256, 128)
CPU_CHUNK_BYTES = 1024


@pytest.fixture
def cpu_kv() -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return [
        torch.randint(-128, 127, (8, page), dtype=torch.int8, generator=gen)
        for page in CPU_PAGES
    ]


def _cpu_kv_caches(kv: list[torch.Tensor]) -> CanonicalKVCaches:
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(t, page) for t, page in zip(kv, CPU_PAGES)],
        group_data_refs=[[CanonicalKVCacheRef(i, p) for i, p in enumerate(CPU_PAGES)]],
    )


@pytest.fixture
def cpu_worker(cpu_kv: list[torch.Tensor]):
    with patch("vllm.v1.kv_offload.obj.cpu_worker.ObjStoreClient", _FakeStore):
        worker = ObjCpuOffloadingWorker(
            kv_caches=_cpu_kv_caches(cpu_kv),
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            chunk_bytes=CPU_CHUNK_BYTES,
            store_config={},
            io_threads=1,
            staging_bytes=CPU_CHUNK_BYTES,  # one slot: a job goes in pieces
            agent_name="test",
        )
    worker._client.staging = worker._staging  # type: ignore[attr-defined]
    yield worker
    worker.shutdown()


def _cpu_ptrs(
    kv: list[torch.Tensor], block_ids: list[int], block_index: int = 0
) -> DevicePointers:
    spec = GPULoadStoreSpec(
        block_ids, group_sizes=[len(block_ids)], block_indices=[block_index]
    )
    return resolve_device_pointers(spec, _cpu_kv_caches(kv))


def _row(kv: list[torch.Tensor], blocks: list[int | None]) -> torch.Tensor:
    """A CPU tier chunk row: each tensor's pages for the chunk's blocks, tensor
    after tensor, then zero padding. None leaves a block's pages zero."""
    parts = [
        t[b] if b is not None else torch.zeros(page, dtype=torch.int8)
        for t, page in zip(kv, CPU_PAGES)
        for b in blocks
    ]
    row = torch.cat(parts)
    return torch.cat([row, torch.zeros(CPU_CHUNK_BYTES - len(row), dtype=torch.int8)])


@pytest.mark.cpu_test
def test_cpu_objects_are_chunk_rows_and_load_back(cpu_worker, cpu_kv):
    """Objects have the GPU worker's layout, so a load restores the blocks."""
    names = ObjLoadStoreSpec(["a", "b"])
    expected = [t[[2, 3, 4, 5]].clone() for t in cpu_kv]

    cpu_worker.submit_store(1, _cpu_ptrs(cpu_kv, [2, 3, 4, 5]), names)
    assert _finish(cpu_worker, 1).success
    objects = cpu_worker._client.objects
    assert torch.equal(objects["a"], _row(cpu_kv, [2, 3]))
    assert torch.equal(objects["b"], _row(cpu_kv, [4, 5]))

    for t in cpu_kv:
        t.zero_()
    cpu_worker.submit_load(2, names, _cpu_ptrs(cpu_kv, [6, 7, 0, 1]))
    assert _finish(cpu_worker, 2).success
    for t, exp in zip(cpu_kv, expected):
        assert torch.equal(t[[6, 7, 0, 1]], exp)


@pytest.mark.cpu_test
def test_cpu_chunk_started_mid_way_keeps_block_offsets(cpu_worker, cpu_kv):
    """A request whose first block is the second of its chunk stores that
    block at the second position of the chunk's pages."""
    cpu_worker.submit_store(
        3, _cpu_ptrs(cpu_kv, [5, 6, 7], 1), ObjLoadStoreSpec(["c", "d"])
    )

    assert _finish(cpu_worker, 3).success
    assert torch.equal(cpu_worker._client.objects["c"], _row(cpu_kv, [None, 5]))
    assert torch.equal(cpu_worker._client.objects["d"], _row(cpu_kv, [6, 7]))


@pytest.mark.cpu_test
def test_cpu_load_of_a_missing_object_fails(cpu_worker, cpu_kv):
    cpu_worker.submit_load(4, ObjLoadStoreSpec(["missing"]), _cpu_ptrs(cpu_kv, [0, 1]))

    assert not _finish(cpu_worker, 4).success


@pytest.mark.cpu_test
def test_cpu_wait_returns_once_the_kv_is_read(cpu_worker, cpu_kv):
    names = ObjLoadStoreSpec(["e", "f"])
    expected = [_row(cpu_kv, [0, 1]), _row(cpu_kv, [2, 3])]
    cpu_worker.submit_store(5, _cpu_ptrs(cpu_kv, [0, 1, 2, 3]), names)
    cpu_worker.wait({5})

    # The KV cache may change once wait() returns: the objects keep the bytes
    # it held at submission.
    for t in cpu_kv:
        t.zero_()
    assert _finish(cpu_worker, 5).success
    assert torch.equal(cpu_worker._client.objects["e"], expected[0])
    assert torch.equal(cpu_worker._client.objects["f"], expected[1])
