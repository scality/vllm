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
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.obj.client import NIXL_WRITE, ObjTransferResult
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.obj.worker import ObjOffloadingWorker, slice_gpu_spec


def _blocks(spec: GPULoadStoreSpec) -> tuple[list[int], list[int], list[int]]:
    return spec.block_ids.tolist(), list(spec.group_sizes), list(spec.block_indices)


@pytest.mark.cpu_test
def test_slice_starts_mid_chunk():
    """The first chunk holds only the blocks past the request's offset in it;
    a slice after it starts on a chunk boundary."""
    spec = GPULoadStoreSpec([10, 11, 12, 13, 14], group_sizes=[5], block_indices=[1])

    assert _blocks(slice_gpu_spec(spec, 2, 0, 1)) == ([10], [1], [1])
    assert _blocks(slice_gpu_spec(spec, 2, 1, 3)) == ([11, 12, 13, 14], [4], [2])


@pytest.mark.cpu_test
def test_slice_numbers_chunks_group_by_group():
    spec = GPULoadStoreSpec(
        [10, 11, 20, 21, 22], group_sizes=[2, 3], block_indices=[0, 0]
    )

    assert _blocks(slice_gpu_spec(spec, 1, 1, 3)) == ([11, 20], [1, 1], [1, 0])
    assert _blocks(slice_gpu_spec(spec, 1, 3, 5)) == ([21, 22], [0, 2], [0, 1])


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
            page_offset=0,
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
