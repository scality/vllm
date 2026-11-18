# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from unittest.mock import patch

import pytest
import torch

from tests.v1.kv_offload.obj.test_worker import _FakeStore
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.hybrid.worker import HybridOffloadingWorker
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.obj.worker import ObjOffloadingWorker


class _RecordingWorker(OffloadingWorker):
    """Finishes each job on demand, with the given success."""

    def __init__(self):
        self.submitted: list[tuple[int, list[int], LoadStoreSpec]] = []
        self.done: list[TransferResult] = []
        self.waited: set[int] = set()

    def submit_store(self, job_id, src_spec, dst_spec):
        self.submitted.append((job_id, src_spec.block_ids.tolist(), dst_spec))
        return True

    def submit_load(self, job_id, src_spec, dst_spec):
        self.submitted.append((job_id, dst_spec.block_ids.tolist(), src_spec))
        return True

    def finish(self, job_id: int, success: bool = True) -> None:
        self.done.append(TransferResult(job_id, success, 10, 0.5))

    def get_finished(self):
        done, self.done = self.done, []
        return done

    def wait(self, job_ids):
        self.waited |= job_ids


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def _split(cpu_chunks, obj_chunks) -> HybridLoadStoreSpec:
    return HybridLoadStoreSpec(
        cpu_chunks,
        CPULoadStoreSpec([10 + c for c in cpu_chunks]),
        obj_chunks,
        ObjLoadStoreSpec([f"o{c}" for c in obj_chunks]),
    )


@pytest.mark.cpu_test
def test_a_split_load_finishes_with_both_parts():
    cpu, obj = _RecordingWorker(), _RecordingWorker()
    worker = HybridOffloadingWorker(cpu, obj, blocks_per_chunk=2)
    worker.submit_load(7, _split([0, 2], [1]), _gpu_spec([0, 1, 2, 3, 4, 5]))

    ((cpu_id, cpu_blocks, cpu_spec),) = cpu.submitted
    ((obj_id, obj_blocks, obj_spec),) = obj.submitted
    assert cpu_blocks == [0, 1, 4, 5] and cpu_spec.block_ids.tolist() == [10, 12]
    assert obj_blocks == [2, 3] and obj_spec.obj_names == ["o1"]

    worker.wait({7})
    assert cpu.waited == {cpu_id} and obj.waited == {obj_id}

    cpu.finish(cpu_id)
    assert worker.get_finished() == []
    obj.finish(obj_id, success=False)
    assert worker.get_finished() == [
        TransferResult(7, False, 20, 0.5, {"cpu": 0.5, "obj": 0.5})
    ]


@pytest.mark.cpu_test
def test_parts_get_increasing_ids_and_single_parts_keep_the_gpu_spec():
    cpu, obj = _RecordingWorker(), _RecordingWorker()
    worker = HybridOffloadingWorker(cpu, obj, blocks_per_chunk=1)
    gpu = _gpu_spec([3, 4])
    worker.submit_store(1, gpu, _split([0, 1], [0, 1]))
    worker.submit_load(2, _split([], [0, 1]), gpu)
    worker.submit_load(3, _split([], []), gpu)

    assert [s[0] for s in cpu.submitted] == [0]
    assert [s[0] for s in obj.submitted] == [1, 2]
    assert obj.submitted[1][1] == [3, 4]
    assert worker.get_finished() == [TransferResult(3, True)]


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="needs a GPU")
def test_store_then_split_load_round_trips():
    page, bpc = 256, 2
    kv = torch.randint(-128, 127, (8, page), dtype=torch.int8, device="cuda")
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(kv, page)],
        group_data_refs=[[CanonicalKVCacheRef(0, page)]],
    )
    cpu = CPUOffloadingWorker(kv_caches, blocks_per_chunk=bpc, num_cpu_chunks=4)
    with patch("vllm.v1.kv_offload.obj.worker.ObjStoreClient", _FakeStore):
        obj = ObjOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=bpc,
            chunk_bytes=4096,
            rank_page_bytes=0,
            canonical_layout=False,
            store_config={},
            io_threads=1,
            staging_bytes=4 * 4096,
            agent_name="test",
        )
    obj._client.staging = obj._staging
    worker = HybridOffloadingWorker(cpu, obj, bpc)

    def finish(job_id: int) -> TransferResult:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for result in worker.get_finished():
                assert result.job_id == job_id
                return result
            time.sleep(0.001)
        raise TimeoutError(job_id)

    try:
        expected = kv[[2, 3, 4, 5]].clone()
        both = HybridLoadStoreSpec(
            [0, 1], CPULoadStoreSpec([0, 1]), [0, 1], ObjLoadStoreSpec(["a", "b"])
        )
        worker.submit_store(1, _gpu_spec([2, 3, 4, 5]), both)
        assert finish(1).success

        kv.zero_()
        split = HybridLoadStoreSpec(
            [1], CPULoadStoreSpec([1]), [0], ObjLoadStoreSpec(["a"])
        )
        worker.submit_load(2, split, _gpu_spec([6, 7, 0, 1]))
        assert finish(2).success
        assert torch.equal(kv[[6, 7, 0, 1]], expected)
    finally:
        worker.shutdown()
