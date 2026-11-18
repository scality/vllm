# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import random
import time
import uuid
from unittest.mock import MagicMock

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

NUM_GPU_BLOCKS = [64]
NUM_CPU_CHUNKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCKS_PER_CHUNK_VALUES = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]
NUM_MAPPINGS_PER_GROUP = [2]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
def test_rocm_cpu_to_gpu_uses_dma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_worker, "HAS_TRITON", True)
    monkeypatch.setattr(gpu_worker.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: True)

    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]]
    assert gpu_worker._select_swap_blocks_fn(refs, gpu_to_cpu=False) is (
        ops.swap_blocks_batch
    )


@pytest.mark.skipif(not gpu_worker.HAS_TRITON, reason="Requires Triton")
def test_unpinned_cpu_to_gpu_uses_dma(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Triton load path dereferences CPU pointers on the GPU, so pageable
    host memory takes the DMA path even for pages where Triton would win."""
    monkeypatch.setattr(gpu_worker.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: False)

    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]]
    assert gpu_worker._select_swap_blocks_fn(refs, gpu_to_cpu=False) is not (
        ops.swap_blocks_batch
    )
    assert gpu_worker._select_swap_blocks_fn(
        refs, gpu_to_cpu=False, host_memory_is_pinned=False
    ) is (ops.swap_blocks_batch)


@pytest.mark.skipif(not gpu_worker.HAS_TRITON, reason="Requires Triton")
def test_prefer_triton_copies_large_pages_to_gpu_with_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large pages take the DMA path by default, and the Triton kernel when it
    is preferred; stores and pageable host memory stay on DMA either way."""
    monkeypatch.setattr(gpu_worker.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: False)

    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=64 * 1024)]]
    assert gpu_worker._select_swap_blocks_fn(refs, gpu_to_cpu=False) is (
        ops.swap_blocks_batch
    )
    assert gpu_worker._select_swap_blocks_fn(
        refs, gpu_to_cpu=False, prefer_triton=True
    ) is not (ops.swap_blocks_batch)
    assert gpu_worker._select_swap_blocks_fn(
        refs, gpu_to_cpu=True, prefer_triton=True
    ) is (ops.swap_blocks_batch)
    assert gpu_worker._select_swap_blocks_fn(
        refs, gpu_to_cpu=False, host_memory_is_pinned=False, prefer_triton=True
    ) is (ops.swap_blocks_batch)


def test_worker_shutdown_releases_region_and_runs_both_handlers() -> None:
    """Both directions drain before the worker releases its shared region."""
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []

    def record_store_shutdown() -> bool:
        calls.append("store")
        return True

    def record_load_shutdown() -> bool:
        calls.append("load")
        return True

    store_handler = MagicMock()
    load_handler = MagicMock()
    store_handler.shutdown.side_effect = record_store_shutdown
    load_handler.shutdown.side_effect = record_load_shutdown

    def record_region_cleanup(**_: bool) -> bool:
        calls.append("region")
        return True

    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    worker.shutdown()

    assert calls == ["store", "load", "region"]
    store_handler.shutdown.assert_called_once_with()
    load_handler.shutdown.assert_called_once_with()
    mmap_region.cleanup.assert_called_once_with()
    assert worker._mmap_region is None


@pytest.mark.parametrize("failing_handler", ["store", "load"])
def test_worker_logs_handler_error_and_cleans_region(
    caplog_vllm, monkeypatch: pytest.MonkeyPatch, failing_handler
) -> None:
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []
    store_handler = MagicMock()
    load_handler = MagicMock()
    if failing_handler == "store":

        def fail_store_shutdown() -> bool:
            calls.append("store")
            raise RuntimeError("transfer did not drain")

        store_handler.shutdown.side_effect = fail_store_shutdown
    else:

        def fail_load_shutdown() -> bool:
            calls.append("load")
            raise RuntimeError("transfer did not drain")

        load_handler.shutdown.side_effect = fail_load_shutdown

    def record_other_shutdown() -> bool:
        calls.append("load" if failing_handler == "store" else "store")
        return True

    if failing_handler == "store":
        load_handler.shutdown.side_effect = record_other_shutdown
    else:
        store_handler.shutdown.side_effect = record_other_shutdown

    def record_device_sync() -> None:
        calls.append("sync")

    monkeypatch.setattr(torch.accelerator, "synchronize", record_device_sync)

    def record_region_cleanup() -> None:
        calls.append("region")

    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    with caplog_vllm.at_level(
        logging.ERROR, logger="vllm.v1.kv_offload.cpu.gpu_worker"
    ):
        worker.shutdown()

    other = "load" if failing_handler == "store" else "store"
    getattr(worker, f"_{other}_handler").shutdown.assert_called_once_with()
    mmap_region.cleanup.assert_called_once_with()
    assert worker._mmap_region is None
    assert (
        f"Failed to shut down {failing_handler} offloading handler" in caplog_vllm.text
    )
    assert calls[-2:] == ["sync", "region"]


def test_handler_shutdown_skips_transfers_after_event_sync_failure() -> None:
    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    failed_event = MagicMock()
    failed_event.synchronize.side_effect = RuntimeError("device lost")
    skipped_event = MagicMock()
    handler._transfers = gpu_worker.deque(
        [
            MagicMock(end_event=failed_event),
            MagicMock(end_event=skipped_event),
        ]
    )
    handler._transfer_events = {1: failed_event, 2: skipped_event}
    handler._stream_pool = [MagicMock()]
    handler._event_pool = [MagicMock()]
    handler._buffer_pool = [(MagicMock(), MagicMock(), MagicMock())]
    handler.src_tensors = [MagicMock()]
    handler.dst_tensors = [MagicMock()]

    with pytest.raises(RuntimeError, match="device lost"):
        handler.shutdown()
    failed_event.synchronize.assert_called_once_with()
    skipped_event.synchronize.assert_not_called()
    assert not handler._transfers
    assert not handler._transfer_events
    assert not handler._stream_pool
    assert not handler._event_pool
    assert not handler._buffer_pool
    assert not handler.src_tensors
    assert not handler.dst_tensors


@pytest.mark.parametrize("device_sync_fails", [False, True])
def test_worker_syncs_before_cleanup_after_handler_failure(
    caplog_vllm, monkeypatch: pytest.MonkeyPatch, device_sync_fails: bool
) -> None:
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []
    store_handler = MagicMock()

    def fail_store_shutdown() -> None:
        raise RuntimeError("device lost")

    store_handler.shutdown.side_effect = fail_store_shutdown
    load_handler = MagicMock()

    def record_device_sync() -> None:
        calls.append("sync")
        if device_sync_fails:
            raise RuntimeError("device lost")

    def record_region_cleanup() -> None:
        calls.append("region")

    monkeypatch.setattr(torch.accelerator, "synchronize", record_device_sync)
    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    with caplog_vllm.at_level(
        logging.WARNING, logger="vllm.v1.kv_offload.cpu.gpu_worker"
    ):
        worker.shutdown()

    assert calls == ["sync", "region"]
    mmap_region.cleanup.assert_called_once_with()
    if device_sync_fails:
        assert "Device sync before mmap cleanup failed" in caplog_vllm.text


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_chunks", NUM_CPU_CHUNKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("use_shared_memory", "replicated_layout"),
    [(False, False), (True, False), (True, True)],
)
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_chunks: int,
    num_tensors: int,
    seed: int,
    device: str,
    use_shared_memory: bool,
    replicated_layout: bool,
) -> None:
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    mmap_region: SharedOffloadRegion | None = None
    if use_shared_memory:
        cpu_page_size = round_up(
            gpu_page_size_bytes * num_tensors * blocks_per_chunk,
            SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        )
        simulated_world_size = 2
        kv_bytes_per_chunk = (
            cpu_page_size if replicated_layout else cpu_page_size * simulated_world_size
        )
        mmap_region = SharedOffloadRegion(
            engine_id=str(uuid.uuid4()),
            num_chunks=num_cpu_chunks,
            rank=0,
            kv_bytes_per_chunk=kv_bytes_per_chunk,
            cpu_page_size=cpu_page_size,
        )

    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_chunks=num_cpu_chunks,
        mmap_region=mmap_region,
    )

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * blocks_per_chunk)
    cpu_chunks = random.sample(range(num_cpu_chunks), num_mappings)

    # expand cpu chunks to gpu-page granularity for uniform comparison:
    # each cpu chunk maps to blocks_per_chunk consecutive sub-blocks
    cpu_chunks_expanded = [
        cpu_chunk * blocks_per_chunk + j
        for cpu_chunk in cpu_chunks
        for j in range(blocks_per_chunk)
    ]

    # maybe skip some GPU blocks to test reading/writing from the middle of a CPU chunk
    blocks_to_skip = blocks_per_chunk - 1
    if blocks_to_skip > 0:
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_chunks_expanded = cpu_chunks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_spec = CPULoadStoreSpec(cpu_chunks)
        dst_to_src = dict(zip(cpu_chunks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_gpu_blocks
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_chunks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_to_src = dict(zip(gpu_blocks, cpu_chunks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # clone src and dst tensors before transfer
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function via public API
    start_time = time.time()
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    # wait for transfer to complete
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            assert finished[0].transfer_size == (
                len(gpu_blocks)
                * sum([x.page_size_bytes for x in handler.layer_refs_per_group[0]])
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        src_view = src_tensor.reshape(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.reshape(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.reshape(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                expected = src_view[src_sub_block]
            else:
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())

    # Drop loop-variable refs so mmap_obj has no exported buffers at cleanup.
    del orig_tensor, tensor, src_tensor, dst_tensor, orig_dst_tensor
    del src_view, dst_view, orig_dst_view, expected

    worker.shutdown()


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings_per_group", NUM_MAPPINGS_PER_GROUP)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_chunks", NUM_CPU_CHUNKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_multi_group(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings_per_group: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_chunks: int,
    seed: int,
    device: str,
) -> None:
    """Test transfers with three KV cache groups:
    - Group 0: aligned transfer with num_mappings_per_group blocks
    - Group 1: zero blocks (empty group)
    - Group 2: unaligned CPU->GPU transfer (logical_offset=blocks_per_chunk-1,
      causing the implementation to skip source sub-blocks) with
      num_mappings_per_group blocks
    """
    set_random_seed(seed)

    # 3 groups, each with 2 tensors
    num_groups = 3
    tensors_per_group = 2
    num_tensors = num_groups * tensors_per_group
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=g * tensors_per_group + i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(tensors_per_group)
        ]
        for g in range(num_groups)
    ]

    canonical_kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors, group_data_refs=kv_cache_groups_data_refs
    )

    worker = CPUOffloadingWorker(
        kv_caches=canonical_kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_chunks=num_cpu_chunks,
    )

    # group 0: aligned, group 1: empty, group 2: unaligned on CPU->GPU
    group_sizes_in_cpu_chunks = [num_mappings_per_group, 0, num_mappings_per_group]

    total_cpu_chunks = sum(group_sizes_in_cpu_chunks)
    total_gpu_blocks_needed = total_cpu_chunks * blocks_per_chunk
    gpu_blocks_all = random.sample(range(num_gpu_blocks), total_gpu_blocks_needed)
    cpu_chunks_all = random.sample(range(num_cpu_chunks), total_cpu_chunks)

    # split gpu blocks / cpu chunks per group
    gpu_blocks_per_group: list[list[int]] = []
    cpu_chunks_per_group: list[list[int]] = []
    gpu_offset = 0
    cpu_offset = 0
    for size in group_sizes_in_cpu_chunks:
        gpu_count = size * blocks_per_chunk
        gpu_blocks_per_group.append(gpu_blocks_all[gpu_offset : gpu_offset + gpu_count])
        cpu_chunks_per_group.append(cpu_chunks_all[cpu_offset : cpu_offset + size])
        gpu_offset += gpu_count
        cpu_offset += size

    # expand cpu chunks to gpu-page granularity
    cpu_chunks_expanded_per_group = [
        [
            cpu_chunk * blocks_per_chunk + j
            for cpu_chunk in cpu_chunks
            for j in range(blocks_per_chunk)
        ]
        for cpu_chunks in cpu_chunks_per_group
    ]

    # skip sub-blocks from group 2 to test unaligned transfers.
    sub_blocks_to_skip = blocks_per_chunk - 1  # e.g. 2 when blocks_per_chunk=3
    if sub_blocks_to_skip > 0:
        gpu_blocks_per_group[2] = gpu_blocks_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]
        cpu_chunks_expanded_per_group[2] = cpu_chunks_expanded_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]

    # build flat gpu_blocks list and group_sizes in GPU blocks
    gpu_blocks: list[int] = []
    group_sizes: list[int] = []
    for gpu_blks in gpu_blocks_per_group:
        gpu_blocks.extend(gpu_blks)
        group_sizes.append(len(gpu_blks))

    # build flat cpu_chunks list
    cpu_chunks = []
    for cpu_chnks in cpu_chunks_per_group:
        cpu_chunks.extend(cpu_chnks)

    # block_indices: only relevant for unaligned transfers
    block_indices: list[int] = [0, 0, sub_blocks_to_skip]

    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        dst_spec = CPULoadStoreSpec(cpu_chunks)
        # per-group mapping: cpu sub-block -> gpu sub-block
        dst_to_src_per_group = [
            dict(zip(expanded, gpu_blks))
            for expanded, gpu_blks in zip(
                cpu_chunks_expanded_per_group, gpu_blocks_per_group
            )
        ]
        num_dst_sub_blocks = num_cpu_chunks * blocks_per_chunk
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_chunks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        # per-group mapping: gpu sub-block -> cpu sub-block
        dst_to_src_per_group = [
            dict(zip(gpu_blks, expanded))
            for gpu_blks, expanded in zip(
                gpu_blocks_per_group, cpu_chunks_expanded_per_group
            )
        ]
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            expected_bytes = sum(
                group_size * sum([x.page_size_bytes for x in data_refs])
                for group_size, data_refs in zip(
                    group_sizes, handler.layer_refs_per_group
                )
            )
            assert finished[0].transfer_size == expected_bytes
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity
    for group_idx, dst_to_src in enumerate(dst_to_src_per_group):
        group_tensor_offset = group_idx * tensors_per_group
        for tensor_idx in range(tensors_per_group):
            src_tensor = handler.src_tensors[group_tensor_offset + tensor_idx]
            dst_tensor = handler.dst_tensors[group_tensor_offset + tensor_idx]
            orig_dst_tensor = orig_dst_tensors[group_tensor_offset + tensor_idx]
            src_view = src_tensor.view(-1, gpu_page_size_bytes)
            dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
            orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
            for dst_sub_block in range(num_dst_sub_blocks):
                src_sub_block = dst_to_src.get(dst_sub_block)
                if src_sub_block is not None:
                    expected = src_view[src_sub_block]
                else:
                    expected = orig_dst_view[dst_sub_block]
                torch.testing.assert_close(
                    dst_view[dst_sub_block].cpu(), expected.cpu()
                )

    worker.shutdown()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="stream ordering test requires a CUDA-like platform",
)
@torch.inference_mode()
def test_load_waits_for_pending_compute_stream_writes(default_vllm_config) -> None:
    """A CPU load must land after pending writes to its GPU destination."""
    device = DEVICES[0]
    page_size_bytes = 128 * 1024
    num_blocks = 64
    loaded_blocks = list(range(32))
    sentinel = 0x5A

    gpu_tensor = torch.zeros(
        (num_blocks, page_size_bytes), dtype=torch.int8, device=device
    )
    loaded_block_ids = torch.tensor(loaded_blocks, dtype=torch.long, device=device)
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(
                    tensor=gpu_tensor, page_size_bytes=page_size_bytes
                )
            ],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size_bytes)]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_chunks=num_blocks,
    )
    worker._load_handler.src_tensors[0].fill_(sentinel)
    expected = torch.full((page_size_bytes,), sentinel, dtype=torch.int8)

    try:
        for trial in range(3):
            gpu_tensor.fill_(0x11)
            torch.accelerator.synchronize()

            # Model a delayed zero of a freshly allocated KV block. Without a
            # compute-stream dependency, the DMA can finish during the sleep
            # and this later fill wipes out the loaded cache contents.
            torch.cuda._sleep(50_000_000)
            gpu_tensor.index_fill_(0, loaded_block_ids, 0)

            assert worker.submit_load(
                trial + 1,
                CPULoadStoreSpec(loaded_blocks),
                GPULoadStoreSpec(
                    loaded_blocks,
                    group_sizes=(len(loaded_blocks),),
                    block_indices=(0,),
                ),
            )
            deadline = time.time() + 10
            finished: list[TransferResult] = []
            while time.time() < deadline and not finished:
                finished = worker.get_finished()
                if not finished:
                    time.sleep(0.001)
            assert finished and finished[0].success, f"load {trial} did not finish"

            torch.accelerator.synchronize()
            for block_id in loaded_blocks:
                torch.testing.assert_close(gpu_tensor[block_id].cpu(), expected)
    finally:
        worker.shutdown()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not gpu_worker.HAS_TRITON,
    reason="Requires CUDA and Triton",
)
def test_staged_loads_match_direct_loads():
    """Loads through the staging buffer write exactly what direct loads write:
    each chunk's pages, laid out in one host row, into its GPU block."""
    pages = [256, 64, 128]
    row = sum(pages)
    num_chunks, num_blocks = 12, 20
    host = torch.randint(-128, 127, (num_chunks * row,), dtype=torch.int8)
    host = host.pin_memory()
    cpu_tensors = []
    offset = 0
    for page in pages:
        cpu_tensors.append(host.as_strided((num_chunks, page), (row, 1), offset))
        offset += page
    refs = [[CanonicalKVCacheRef(i, page) for i, page in enumerate(pages)]]
    chunks = [3, 0, 7, 11, 5, 1, 9, 2, 8, 6]
    blocks = [10, 2, 19, 4, 6, 0, 13, 7, 15, 1]

    results = []
    # 16 slots: a piece of 8 chunks scattered by the Triton kernel, then 2.
    for staging_bytes in (0, 16 * row):
        gpu_tensors = [
            torch.zeros((num_blocks, page), dtype=torch.int8, device="cuda")
            for page in pages
        ]
        handler = gpu_worker.SingleDirectionOffloadingHandler(
            list(gpu_tensors),
            list(cpu_tensors),
            1,
            refs,
            gpu_to_cpu=False,
            staging_bytes=staging_bytes,
        )
        assert (handler._staged is not None) == bool(staging_bytes)
        handler.transfer_async(
            1,
            CPULoadStoreSpec(chunks),
            GPULoadStoreSpec(blocks, group_sizes=[len(blocks)], block_indices=[0]),
        )
        torch.accelerator.synchronize()
        (result,) = handler.get_finished()
        assert result.success and result.transfer_size == len(chunks) * row
        handler.shutdown()
        results.append([t.cpu() for t in gpu_tensors])

    for direct, staged, cpu_tensor in zip(*results, cpu_tensors):
        assert torch.equal(direct, staged)
        for chunk, block in zip(chunks, blocks):
            assert torch.equal(staged[block], cpu_tensor[chunk])


def _fanout_worker(rank: int, port: int, result_queue) -> None:
    import torch.distributed as dist

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )

    def all_gather(obj):
        gathered = [None, None]
        dist.all_gather_object(gathered, obj)
        return gathered

    pages = [256, 64, 128]
    row = sum(pages)
    num_chunks, num_blocks = 12, 20
    # Every worker reads the same host rows, as with the replicated layout.
    gen = torch.Generator().manual_seed(0)
    host = torch.randint(
        -128, 127, (num_chunks * row,), dtype=torch.int8, generator=gen
    ).pin_memory()
    cpu_tensors, offset = [], 0
    for page in pages:
        cpu_tensors.append(host.as_strided((num_chunks, page), (row, 1), offset))
        offset += page
    gpu_tensors = [
        torch.zeros((num_blocks, page), dtype=torch.int8, device="cuda")
        for page in pages
    ]
    handler = gpu_worker.SingleDirectionOffloadingHandler(
        list(gpu_tensors),
        list(cpu_tensors),
        1,
        [[CanonicalKVCacheRef(i, page) for i, page in enumerate(pages)]],
        gpu_to_cpu=False,
        staging_bytes=16 * row,
        fanout=gpu_worker.FanOut(rank, 2, all_gather),
    )
    chunks = [3, 0, 7, 11, 5, 1, 9, 2, 8, 6]
    blocks = [10, 2, 19, 4, 6, 0, 13, 7, 15, 1]
    handler.transfer_async(
        1,
        CPULoadStoreSpec(chunks),
        GPULoadStoreSpec(blocks, group_sizes=[len(blocks)], block_indices=[0]),
    )
    torch.accelerator.synchronize()
    (result,) = handler.get_finished()
    dist.barrier()  # every worker has written its chunks into both caches
    loaded = all(
        torch.equal(gpu[block].cpu(), cpu[chunk])
        for gpu, cpu in zip(gpu_tensors, cpu_tensors)
        for chunk, block in zip(chunks, blocks)
    )
    result_queue.put((rank, result.success, result.transfer_size, loaded))
    dist.barrier()
    handler.shutdown()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
def test_fanout_loads_fill_every_worker():
    """With fan-out, each worker loads half of the chunks from host memory and
    writes them into both workers' KV caches: each cache ends up with all."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(_fanout_worker, args=(get_open_port(), result_queue), nprocs=2)
    results = sorted(result_queue.get() for _ in range(2))
    row = 256 + 64 + 128
    assert all(success and loaded for _, success, _, loaded in results)
    # Each worker moved its 5 chunks into 2 caches.
    assert [size for _, _, size, _ in results] == [5 * row * 2] * 2


def test_staged_loads_take_canonical_pages_that_are_the_gpu_pages():
    """A canonical page that is the GPU page byte for byte (MLA) loads like a
    direct one; one assembled from fragments (sharded heads) does not."""
    from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

    def ref(runs, canonical_size=256):
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=canonical_size,
            local_page_size_bytes=256,
            runs=tuple(runs),
            num_writers=2,
            writer_index=0,
            parallelism_agnostic=True,
        )
        return CanonicalKVCacheRef(0, 256, mapping)

    assert gpu_worker._is_identity(ref([CopyRun(0, 0, 256, 1, 256, 256)]))
    assert gpu_worker._is_identity(ref([CopyRun(0, 0, 64, 4, 64, 64)]))
    assert not gpu_worker._is_identity(
        ref([CopyRun(0, 128, 64, 2, 64, 256)], canonical_size=512)
    )
    assert not gpu_worker._is_identity(CanonicalKVCacheRef(0, 256))
