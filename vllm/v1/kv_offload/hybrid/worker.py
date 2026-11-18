# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of offloading KV to host memory and an object store at once:
each part of a transfer runs as a job of its own medium's worker."""

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    DevicePointers,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.pointer_utils import select_device_chunks

logger = init_logger(__name__)


@dataclass
class _Transfer:
    job_id: int
    parts_left: int
    success: bool = True
    size: int = 0
    time: float = 0.0


class HybridOffloadingWorker(OffloadingWorker):
    def __init__(
        self,
        cpu: OffloadingWorker,
        obj: OffloadingWorker,
        blocks_per_chunk: int,
    ):
        self._workers = (cpu, obj)
        self._blocks_per_chunk = blocks_per_chunk
        # Part job ids increase in submission order, the same on every rank,
        # as the object store worker's cross-rank fences require.
        self._next_part_id = 0
        self._parts: tuple[dict[int, _Transfer], dict[int, _Transfer]] = ({}, {})
        self._part_ids: dict[int, list[tuple[int, int]]] = {}
        self._finished: list[TransferResult] = []

    def _submit(
        self,
        job_id: int,
        is_store: bool,
        device_ptrs: DevicePointers,
        spec: LoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        assert isinstance(spec, HybridLoadStoreSpec)
        parts = [
            (medium, chunks, sub_spec)
            for medium, (chunks, sub_spec) in enumerate(
                ((spec.cpu_chunks, spec.cpu), (spec.obj_chunks, spec.obj))
            )
            if chunks
        ]
        if not parts:
            self._finished.append(TransferResult(job_id, True))
            return True
        transfer = _Transfer(job_id, len(parts))
        self._part_ids[job_id] = []
        for medium, chunks, sub_spec in parts:
            part_id = self._next_part_id
            self._next_part_id += 1
            self._parts[medium][part_id] = transfer
            self._part_ids[job_id].append((medium, part_id))
            part_ptrs = (
                device_ptrs
                if len(parts) == 1
                else select_device_chunks(device_ptrs, self._blocks_per_chunk, chunks)
            )
            worker = self._workers[medium]
            if is_store:
                ok = worker.submit_store(part_id, part_ptrs, sub_spec)
            else:
                ok = worker.submit_load(part_id, sub_spec, part_ptrs, expedited)
            assert ok
        return True

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id, True, device_ptrs, dst_spec)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        device_ptrs: DevicePointers,
        expedited: bool = False,
    ) -> bool:
        return self._submit(job_id, False, device_ptrs, src_spec, expedited)

    def get_finished(self) -> list[TransferResult]:
        finished, self._finished = self._finished, []
        for medium, worker in enumerate(self._workers):
            for result in worker.get_finished():
                transfer = self._parts[medium].pop(result.job_id)
                transfer.success &= result.success
                transfer.size += result.transfer_size or 0
                transfer.time = max(transfer.time, result.transfer_time or 0.0)
                logger.debug(
                    "job %d: %s part done in %.1f ms (%d bytes)",
                    transfer.job_id,
                    ("cpu", "obj")[medium],
                    (result.transfer_time or 0.0) * 1e3,
                    result.transfer_size or 0,
                )
                transfer.parts_left -= 1
                if transfer.parts_left == 0:
                    del self._part_ids[transfer.job_id]
                    finished.append(
                        TransferResult(
                            transfer.job_id,
                            transfer.success,
                            transfer.size or None,
                            transfer.time or None,
                        )
                    )
        return finished

    def wait(self, job_ids: set[int]) -> None:
        for medium, worker in enumerate(self._workers):
            part_ids = {
                part_id
                for job_id in job_ids
                for part_medium, part_id in self._part_ids.get(job_id, ())
                if part_medium == medium
            }
            if part_ids:
                worker.wait(part_ids)

    def shutdown(self) -> None:
        for worker in self._workers:
            worker.shutdown()
