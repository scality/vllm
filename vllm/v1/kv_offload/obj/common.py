# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
from collections.abc import Sequence

from vllm.v1.kv_offload.base import LoadStoreSpec


class ObjLoadStoreSpec(LoadStoreSpec):
    """The objects holding the chunks of a transfer, one per chunk, in the
    order of the chunks of the GPU spec."""

    def __init__(self, obj_names: list[str]):
        self.obj_names = obj_names

    def __repr__(self) -> str:
        return f"ObjLoadStoreSpec({len(self.obj_names)} objects)"

    def select(self, indices: Sequence[int]) -> "ObjLoadStoreSpec":
        return ObjLoadStoreSpec([self.obj_names[i] for i in indices])


def owner_of(obj_name: str, world_size: int) -> int:
    """The worker that moves this object, the same in every process. CRC32 is
    linear: names differing in a digit would share an owner."""
    digest = hashlib.blake2b(obj_name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % world_size
