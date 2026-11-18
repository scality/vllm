# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from vllm.v1.kv_offload.base import LoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec


class HybridLoadStoreSpec(LoadStoreSpec):
    """The chunks of one transfer that host memory and the object store move.

    cpu_chunks and obj_chunks are increasing positions of chunks in the
    transfer, in the order of cpu and obj. A load reads each chunk from one
    of them; a store may write a chunk to both.
    """

    def __init__(
        self,
        cpu_chunks: list[int],
        cpu: CPULoadStoreSpec,
        obj_chunks: list[int],
        obj: ObjLoadStoreSpec,
    ):
        assert len(cpu_chunks) == len(cpu.block_ids)
        assert len(obj_chunks) == len(obj.obj_names)
        self.cpu_chunks = cpu_chunks
        self.cpu = cpu
        self.obj_chunks = obj_chunks
        self.obj = obj

    def __repr__(self) -> str:
        return (
            f"HybridLoadStoreSpec(cpu={len(self.cpu_chunks)}, "
            f"obj={len(self.obj_chunks)})"
        )

    def select(self, indices: Sequence[int]) -> "HybridLoadStoreSpec":
        cpu_at = {chunk: row for row, chunk in enumerate(self.cpu_chunks)}
        obj_at = {chunk: row for row, chunk in enumerate(self.obj_chunks)}
        cpu_chunks, cpu_rows, obj_chunks, obj_rows = [], [], [], []
        for position, chunk in enumerate(indices):
            if chunk in cpu_at:
                cpu_chunks.append(position)
                cpu_rows.append(cpu_at[chunk])
            if chunk in obj_at:
                obj_chunks.append(position)
                obj_rows.append(obj_at[chunk])
        return HybridLoadStoreSpec(
            cpu_chunks, self.cpu.select(cpu_rows), obj_chunks, self.obj.select(obj_rows)
        )
