# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Access to the KV caches of the other workers of a node, for loads that one
worker writes into every worker's KV cache."""

import ctypes
import secrets
import time
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any, cast

import numpy as np
import torch

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Returns the objects every worker passed, in rank order.
AllGather = Callable[[Any], list[Any]]


class PeerTensor:
    """Another worker's KV cache tensor, mapped into this process: what the
    copy handler reads of a tensor, without a torch tensor, which would open a
    CUDA context on the other worker's GPU."""

    dtype = torch.int8
    ndim = 2
    is_cuda = True
    is_xpu = False

    def __init__(self, ptr: int, shape: tuple[int, int], row_stride: int):
        self._ptr = ptr
        self.shape = shape
        self._row_stride = row_stride
        self.device = torch.device("cuda", torch.accelerator.current_device_index())

    def data_ptr(self) -> int:
        return self._ptr

    def stride(self, dim: int) -> int:
        return (self._row_stride, 1)[dim]


def _allocation_base(ptr: int) -> int:
    """Start of the CUDA allocation holding ptr: CUDA IPC shares whole
    allocations."""
    libcuda = ctypes.CDLL("libcuda.so.1")
    base = ctypes.c_uint64()
    size = ctypes.c_size_t()
    result = libcuda.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)
    )
    if result != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed ({result})")
    return base.value


def _ipc_export(
    lib: CudaRTLibrary, tensor: torch.Tensor
) -> tuple[bytes, int, tuple[int, int], int]:
    assert tensor.dtype == torch.int8 and tensor.ndim == 2
    base = _allocation_base(tensor.data_ptr())
    handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(base))
    return (
        bytes(handle),
        tensor.data_ptr() - base,
        (tensor.shape[0], tensor.shape[1]),
        tensor.stride(0),
    )


def share_tensors(
    tensors: list[torch.Tensor], rank: int, all_gather: AllGather
) -> list[list[torch.Tensor]]:
    """Every worker's tensors (int8, 2D), in rank order: this worker's own,
    the others' as PeerTensors mapped into this process through CUDA IPC."""
    lib = CudaRTLibrary()
    exports = all_gather([_ipc_export(lib, t) for t in tensors])
    # One mapping per allocation: tensors may share one.
    bases: dict[bytes, int] = {}
    peers: list[list[torch.Tensor]] = []
    for peer, peer_exports in enumerate(exports):
        if peer == rank:
            peers.append(tensors)
            continue
        mapped: list[torch.Tensor] = []
        for handle, offset, shape, row_stride in peer_exports:
            if handle not in bases:
                ipc_handle = cudaIpcMemHandle_t.from_buffer_copy(handle)
                base = lib.cudaIpcOpenMemHandle(ipc_handle).value
                assert base is not None
                bases[handle] = base
            ptr = bases[handle] + offset
            mapped.append(cast(torch.Tensor, PeerTensor(ptr, shape, row_stride)))
        peers.append(mapped)
    return peers


class CrossRankFence:
    """Lets a worker wait, on its own stream, until another worker's KV cache
    is ready for a load job.

    Each worker records an interprocess event on its compute stream when it
    receives a load job, then publishes the job id in shared memory. A waiter
    seeing a published id at least the job's waits on that event: a later
    record covers the earlier work too, as job ids increase.
    """

    def __init__(self, rank: int, world_size: int, all_gather: AllGather):
        self._rank = rank
        self._event = torch.cuda.Event(interprocess=True)
        name = all_gather(
            f"vllm_obj_fence_{secrets.token_hex(8)}" if rank == 0 else None
        )[0]
        if rank == 0:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=8 * world_size
            )
        all_gather(None)
        if rank != 0:
            self._shm = shared_memory.SharedMemory(name=name)
        self._ready = np.ndarray((world_size,), dtype=np.int64, buffer=self._shm.buf)
        if rank == 0:
            self._ready[:] = -1
        all_gather(None)
        device = torch.accelerator.current_device_index()
        handles = all_gather(self._event.ipc_handle())
        self._events = [
            self._event
            if peer == rank
            else torch.cuda.Event.from_ipc_handle(device, handle)
            for peer, handle in enumerate(handles)
        ]

    def publish(self, job_id: int) -> None:
        """This worker's KV cache is ready for job_id once its current stream
        has reached this point."""
        self._event.record()
        self._ready[self._rank] = job_id

    def wait(self, job_id: int, stream: torch.cuda.Stream, timeout_s: float) -> None:
        """Make stream wait until every worker's KV cache is ready for job_id.

        Raises:
            TimeoutError: a worker did not publish job_id in time.

        """
        deadline = time.monotonic() + timeout_s
        for peer, event in enumerate(self._events):
            while self._ready[peer] < job_id:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"worker {peer} did not publish load job {job_id}"
                    )
                time.sleep(1e-5)
            stream.wait_event(event)

    def close(self) -> None:
        del self._ready
        self._shm.close()
        if self._rank == 0:
            self._shm.unlink()
