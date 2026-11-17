# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
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
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.obj.client import NIXL_WRITE, ObjTransferResult
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec, owner_of
from vllm.v1.kv_offload.obj.worker import ObjOffloadingWorker, select_gpu_chunks


def _blocks(spec: GPULoadStoreSpec) -> tuple[list[int], list[int], list[int]]:
    return spec.block_ids.tolist(), list(spec.group_sizes), list(spec.block_indices)


@pytest.mark.cpu_test
def test_slice_starts_mid_chunk():
    """The first chunk holds only the blocks past the request's offset in it;
    a slice after it starts on a chunk boundary."""
    spec = GPULoadStoreSpec([10, 11, 12, 13, 14], group_sizes=[5], block_indices=[1])

    assert _blocks(select_gpu_chunks(spec, 2, [0])) == ([10], [1], [1])
    assert _blocks(select_gpu_chunks(spec, 2, [1, 2])) == ([11, 12, 13, 14], [4], [2])


@pytest.mark.cpu_test
def test_slice_numbers_chunks_group_by_group():
    spec = GPULoadStoreSpec(
        [10, 11, 20, 21, 22], group_sizes=[2, 3], block_indices=[0, 0]
    )

    assert _blocks(select_gpu_chunks(spec, 1, [1, 2])) == ([11, 20], [1, 1], [1, 0])
    assert _blocks(select_gpu_chunks(spec, 1, [3, 4])) == ([21, 22], [0, 2], [0, 1])


@pytest.mark.cpu_test
def test_select_skips_chunks_owned_by_other_workers():
    """The chunks a worker owns need not be consecutive; their blocks fill
    its slots in order."""
    spec = GPULoadStoreSpec([10, 11, 12, 13, 14], group_sizes=[5], block_indices=[1])

    assert _blocks(select_gpu_chunks(spec, 2, [0, 2])) == ([10, 13, 14], [3], [1])


@pytest.mark.cpu_test
def test_every_chunk_has_one_owner_in_every_process():
    names = [f"model/{i:03x}/{i}.bin" for i in range(64)]
    owners = [owner_of(n, 8) for n in names]

    assert owners == [owner_of(n, 8) for n in names]
    assert set(owners) == set(range(8))


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
        torch.accelerator.synchronize()
        success = True
        for slot, name in zip(slot_ids.tolist(), obj_names):
            if op == NIXL_WRITE:
                self.objects[name] = self.staging[slot].clone()
            elif name in self.objects:
                self.staging[slot].copy_(self.objects[name])
            else:
                success = False
        torch.accelerator.synchronize()
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


@pytest.fixture
def worker(kv: torch.Tensor):
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
            staging_bytes=CHUNK_BYTES,  # one slot: a job goes in pieces
            agent_name="test",
        )
    worker._client.staging = worker._staging
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


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def test_store_then_load_round_trips(worker, kv):
    """Each object is a CPU tier chunk row: the chunk's pages, then padding."""
    names = ObjLoadStoreSpec(["a", "b"])
    expected = kv[[2, 3, 4, 5]].clone()

    worker.submit_store(1, _gpu_spec([2, 3, 4, 5]), names)
    assert _finish(worker, 1).success
    objects = worker._client.objects
    assert torch.equal(objects["a"][: 2 * PAGE].view(2, PAGE), expected[:2])
    assert torch.equal(objects["b"][: 2 * PAGE].view(2, PAGE), expected[2:])

    kv.zero_()
    worker.submit_load(2, names, _gpu_spec([6, 7, 0, 1]))
    assert _finish(worker, 2).success
    assert torch.equal(kv[[6, 7, 0, 1]], expected)


def test_load_of_a_missing_object_fails(worker, kv):
    worker.submit_load(3, ObjLoadStoreSpec(["missing"]), _gpu_spec([0, 1]))

    assert not _finish(worker, 3).success


def test_wait_returns_once_the_kv_is_read(worker):
    worker.submit_store(4, _gpu_spec([0, 1, 2, 3]), ObjLoadStoreSpec(["c", "d"]))
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
    worker._client.staging = worker._staging
    names = ObjLoadStoreSpec([f"chunk{i}" for i in range(4)])
    expected = kv[:8].clone()

    worker.submit_store(1, _gpu_spec(list(range(8))), names)
    stored = _finish(worker, 1).success
    dist.barrier()
    kv[8:] = 0
    torch.accelerator.synchronize()
    dist.barrier()
    worker.submit_load(2, names, _gpu_spec(list(range(8, 16))))
    loaded = _finish(worker, 2).success
    dist.barrier()  # every owner has written into every worker's KV cache
    result_queue.put(
        (
            rank,
            stored,
            loaded,
            set(worker._client.objects),
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
    worker._client.staging = worker._staging
    # One block per chunk takes the fast load path.
    assert (worker._fast_loads is not None) == (bpc == 1)
    names = ObjLoadStoreSpec([f"chunk{i}" for i in range(8 // bpc)])
    expected = kv[:8].clone()

    worker.submit_store(1, _gpu_spec(list(range(8))), names)
    worker.wait({1})  # every owner has read this worker's blocks
    kv[:8] = 0  # so the store cannot have read them late
    stored = _finish(worker, 1).success
    dist.barrier()
    worker.submit_load(2, names, _gpu_spec(list(range(8, 16))))
    loaded = _finish(worker, 2).success
    dist.barrier()  # every owner has written into every worker's KV cache
    # Bytes: a tensor would go through shared memory gone with this process.
    objects = {n: o.cpu().numpy().tobytes() for n, o in worker._client.objects.items()}
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
    worker._client.staging = worker._staging
    try:
        assert worker._fast_loads is not None
        names = ObjLoadStoreSpec([f"c{i}" for i in range(3)])
        expected = kv[[1, 2, 3]].clone()
        worker.submit_store(1, _gpu_spec([1, 2, 3]), names)
        assert _finish(worker, 1).success
        kv.zero_()
        worker.submit_load(2, names, _gpu_spec([7, 0, 5]))
        assert _finish(worker, 2).success
        assert torch.equal(kv[[7, 0, 5]], expected)
    finally:
        worker.shutdown()
