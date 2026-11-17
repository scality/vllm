# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for resolving device (gpu,tpu,etc) pointers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    DevicePointerGroupInfo,
    DevicePointers,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches,
        GPULoadStoreSpec,
    )


def iter_pointer_groups(
    device_ptrs: DevicePointers,
    blocks_per_chunk: int,
) -> Iterator[DevicePointerGroupInfo]:
    """Iterate non-empty groups with pre-computed chunk/skip metadata."""
    dev_ptr_offset = 0
    for group_idx, (group_size, n_data_refs, block_idx) in enumerate(
        zip(
            device_ptrs.group_block_counts,
            device_ptrs.group_data_ref_counts,
            device_ptrs.block_indices,
        )
    ):
        if group_size == 0:
            continue
        skip = block_idx % blocks_per_chunk
        n_chunks = cdiv(group_size + skip, blocks_per_chunk)
        yield DevicePointerGroupInfo(
            group_idx=group_idx,
            group_size=group_size,
            n_data_refs=n_data_refs,
            skip=skip,
            n_chunks=n_chunks,
            dev_ptr_offset=dev_ptr_offset,
        )
        dev_ptr_offset += group_size * n_data_refs


def resolve_device_pointers(
    device_spec: GPULoadStoreSpec,
    kv_caches: CanonicalKVCaches,
) -> DevicePointers:
    """Convert logical block IDs to device memory pointers.
    GPU blocks_per_chunk is always 1, so resolution is a direct
    vectorized lookup: base_ptr + block_id * row_stride.
    """
    block_ids = device_spec.block_ids
    group_sizes = device_spec.group_sizes
    block_indices = device_spec.block_indices
    group_data_refs = kv_caches.group_data_refs

    assert len(group_sizes) == len(group_data_refs)
    assert len(block_indices) == len(group_sizes)

    num_copy_ops = sum(gs * len(refs) for gs, refs in zip(group_sizes, group_data_refs))

    ptrs = np.empty(num_copy_ops, dtype=np.uint64)
    sizes = np.empty(num_copy_ops, dtype=np.uint64)

    blk_offset = 0
    op_idx = 0

    for group_size, data_refs in zip(group_sizes, group_data_refs):
        if group_size == 0:
            continue
        group_block_ids = block_ids[blk_offset : blk_offset + group_size]
        group_block_ids_u64 = group_block_ids.astype(np.uint64)
        for data_ref in data_refs:
            tensor = kv_caches.tensors[data_ref.tensor_idx].tensor
            base_ptr = np.uint64(tensor.data_ptr())
            row_stride = np.uint64(tensor.stride(0))
            end_idx = op_idx + group_size
            ptrs[op_idx:end_idx] = base_ptr + group_block_ids_u64 * row_stride
            sizes[op_idx:end_idx] = data_ref.page_size_bytes
            op_idx = end_idx
        blk_offset += group_size

    assert blk_offset == len(block_ids)
    assert op_idx == num_copy_ops

    return DevicePointers(
        ptrs=ptrs,
        sizes=sizes,
        group_block_counts=tuple(group_sizes),
        group_data_ref_counts=tuple(len(refs) for refs in group_data_refs),
        block_indices=tuple(block_indices),
    )


def select_device_chunks(
    device_ptrs: DevicePointers, blocks_per_chunk: int, chunks: Sequence[int]
) -> DevicePointers:
    """The pointers of these chunks of a transfer, in increasing order; chunks
    are numbered group by group as iter_pointer_groups() numbers them.

    Only a transfer's first chunk in a group can start mid-chunk and only its
    last can end mid-chunk, so the blocks of the selected chunks of a group
    fill their slots in order, as those of one transfer do."""
    ptrs: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    group_block_counts: list[int] = []
    block_indices: list[int] = []
    chunk_offset = 0
    dev_ptr_offset = 0
    i = 0
    for group_size, n_data_refs, block_idx in zip(
        device_ptrs.group_block_counts,
        device_ptrs.group_data_ref_counts,
        device_ptrs.block_indices,
    ):
        skip = block_idx % blocks_per_chunk
        n_chunks = cdiv(group_size + skip, blocks_per_chunk) if group_size else 0
        ranges: list[tuple[int, int]] = []
        while i < len(chunks) and chunks[i] < chunk_offset + n_chunks:
            chunk = chunks[i] - chunk_offset
            lo = max(chunk * blocks_per_chunk - skip, 0)
            hi = min((chunk + 1) * blocks_per_chunk - skip, group_size)
            ranges.append((lo, hi))
            i += 1
        for d in range(n_data_refs):
            ref_offset = dev_ptr_offset + d * group_size
            for lo, hi in ranges:
                ptrs.append(device_ptrs.ptrs[ref_offset + lo : ref_offset + hi])
                sizes.append(device_ptrs.sizes[ref_offset + lo : ref_offset + hi])
        group_block_counts.append(sum(hi - lo for lo, hi in ranges))
        block_indices.append(block_idx + (ranges[0][0] if ranges else 0))
        chunk_offset += n_chunks
        dev_ptr_offset += group_size * n_data_refs
    assert i == len(chunks)
    empty = np.empty(0, dtype=np.uint64)
    return DevicePointers(
        ptrs=np.concatenate(ptrs) if ptrs else empty,
        sizes=np.concatenate(sizes) if sizes else empty,
        group_block_counts=tuple(group_block_counts),
        group_data_ref_counts=device_ptrs.group_data_ref_counts,
        block_indices=tuple(block_indices),
    )


def remap_device_pointers(
    device_ptrs: DevicePointers,
    data_ref_tensors: Sequence[Sequence[int]],
    sources: Sequence[Sequence[tuple[int, int]]],
    own_bases: Sequence[int],
    bases: Sequence[Sequence[int]],
) -> DevicePointers:
    """Pointers of the same blocks in other copies of the tensors, which hold
    them at the same offsets (e.g. other workers' KV caches mapped through
    CUDA IPC).

    Args:
        device_ptrs: Pointers into the tensors at own_bases.
        data_ref_tensors: Per group, the tensor index of each data ref of
            device_ptrs.
        sources: Per group, for each data ref of the result, the copy it is
            in and the data ref of device_ptrs it stands for.
        own_bases: Base address of each tensor device_ptrs points into.
        bases: Per copy, the base address of each tensor.

    """
    ptrs: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    dev_ptr_offset = 0
    for group_size, tensor_ids, group_sources in zip(
        device_ptrs.group_block_counts, data_ref_tensors, sources
    ):
        for copy_idx, d in group_sources:
            t_idx = tensor_ids[d]
            ref_offset = dev_ptr_offset + d * group_size
            own = device_ptrs.ptrs[ref_offset : ref_offset + group_size]
            ptrs.append(
                own - np.uint64(own_bases[t_idx]) + np.uint64(bases[copy_idx][t_idx])
            )
            sizes.append(device_ptrs.sizes[ref_offset : ref_offset + group_size])
        dev_ptr_offset += group_size * len(tensor_ids)
    empty = np.empty(0, dtype=np.uint64)
    return DevicePointers(
        ptrs=np.concatenate(ptrs) if ptrs else empty,
        sizes=np.concatenate(sizes) if sizes else empty,
        group_block_counts=device_ptrs.group_block_counts,
        group_data_ref_counts=tuple(len(group) for group in sources),
        block_indices=device_ptrs.block_indices,
    )
