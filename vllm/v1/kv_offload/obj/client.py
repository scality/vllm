# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL client moving fixed-size slots of local memory to and from objects."""

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from vllm.distributed.nixl_utils import NixlWrapper as nixl_agent
from vllm.distributed.nixl_utils import nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.kv_offload.obj.config import ObjStoreConfig

if TYPE_CHECKING:
    from nixl._api import nixl_prepped_dlist_handle, nixl_xfer_handle

logger = init_logger(__name__)

NIXL_WRITE = "WRITE"
NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

# Maximum size of a single NIXL memory registration. cuObject (used by the
# accelerated OBJ engines) rejects any registration larger than 4 GiB, so local
# memory is registered in chunks aligned to slot boundaries. The default S3
# engine has no such limit but tolerates the chunked registrations.
MAX_REG_BYTES: int = 4 * 1024**3

# Fields for NIXL OBJ descriptors: (addr, len, dev_id, obj_key).
# For existence probes addr and len are placeholders — no data is read.
# dev_id=0 is reserved for probes; transfers start from 1.
_PROBE_ADDR: int = 0
_PROBE_LEN: int = 1
_PROBE_DEV_ID: int = 0


class _Transfer(NamedTuple):
    xfer_handle: "nixl_xfer_handle"
    files_desc: object
    obj_handle: "nixl_prepped_dlist_handle"


class ObjTransferResult(NamedTuple):
    job_id: int
    success: bool
    # Seconds, from NIXL telemetry; None when unavailable.
    transfer_time: float | None = None


class ObjStoreClient:
    """Moves slots of one registered local buffer (host or GPU memory) to and
    from objects, through a NIXL backend exposing object segments.

    Transfers are asynchronous: ``submit`` posts one, ``poll`` returns those
    that ended. Nothing here blocks except ``drain``.
    """

    def __init__(self, agent_name: str, store_config: dict, io_threads: int):
        """Args:
        agent_name: NIXL agent name.
        store_config: Object store connection parameters (see ObjStoreConfig).
        io_threads: Number of NIXL I/O threads.

        Raises:
            ValueError: The backend is missing or has no object segments.
            RuntimeError: The object store cannot be reached.

        """
        self._agent_name = agent_name
        # Telemetry supplies the per-transfer duration in ObjTransferResult.
        agent_config = nixl_agent_config(backends=[], capture_telemetry=True)
        self._agent = nixl_agent(agent_name, agent_config)
        obj_config = ObjStoreConfig(**store_config)
        self.backend = obj_config.backend
        if "OBJ_SEG" not in self._agent.get_plugin_mem_types(self.backend):
            raise ValueError(
                f"NIXL backend {self.backend!r} is not available or does not "
                f"support object segments (OBJ_SEG). Available plugins: "
                f"{', '.join(self._agent.get_plugin_list())}."
            )
        self.is_accelerated = obj_config.accelerated
        params = {**obj_config.to_nixl_params(), "num_threads": str(io_threads)}
        self._agent.create_backend(self.backend, params)

        self._transfers: dict[int, _Transfer] = {}
        # Results awaiting the next poll(): submission-time failures and
        # completions found by drain().
        self._pending_results: list[ObjTransferResult] = []
        self._local_regs: list = []
        self._local_handle: nixl_prepped_dlist_handle | None = None
        self._slot_bytes: int = 0
        # The OBJ backend maps devId -> obj_key; dev_id=0 is reserved for probes.
        self._next_obj_dev_id: int = 1

        self._probe_connectivity(obj_config)

    @property
    def num_in_flight(self) -> int:
        return len(self._transfers)

    def _probe_connectivity(self, obj_config: ObjStoreConfig) -> None:
        """Query a key that never exists: an answer either way proves the store
        is reachable; an exception means misconfigured store params."""
        if self.backend != "OBJ":
            hint = f"endpoint_override and extra_params of backend {self.backend}"
        elif obj_config.accelerated:
            hint = "type and endpoint_override"
        else:
            hint = (
                "bucket, endpoint_override, and scheme. If using explicit "
                "credentials verify access_key and secret_key; otherwise ensure "
                "the AWS SDK default credential chain is configured (IAM role, "
                "env vars, credential file)"
            )
        try:
            self.query_existence(["__nixl_probe__/connectivity_test"])
            logger.info("Object store connectivity probe succeeded")
        except Exception as e:
            raise RuntimeError(
                f"Object store connectivity probe failed — check {hint}. Error: {e}"
            ) from e

    def register_local(
        self,
        base_addr: int,
        num_slots: int,
        slot_bytes: int,
        mem_type: Literal["DRAM", "VRAM"] = "DRAM",
        dev_id: int = 0,
    ) -> None:
        """Register the local buffer whose slots transfers read and write.

        Args:
            base_addr: Address of slot 0.
            num_slots: Number of contiguous slots.
            slot_bytes: Size of a slot, and of every object.
            mem_type: "DRAM" for host memory, "VRAM" for GPU memory.
            dev_id: GPU ordinal for VRAM; 0 for DRAM.

        """
        assert self._local_handle is None, "local buffer already registered"
        # Registrations stay below MAX_REG_BYTES and aligned to slots, so no
        # slot straddles two (NIXL requires each transfer descriptor to be
        # covered by a single registration).
        if slot_bytes > MAX_REG_BYTES:
            raise RuntimeError(
                f"KV block size ({slot_bytes} bytes) exceeds the maximum NIXL "
                f"registration size ({MAX_REG_BYTES} bytes). Reduce block_size."
            )
        slots_per_reg = MAX_REG_BYTES // slot_bytes
        for start in range(0, num_slots, slots_per_reg):
            n = min(slots_per_reg, num_slots - start)
            reg = self._agent.register_memory(
                [(base_addr + start * slot_bytes, n * slot_bytes, dev_id, "")],
                mem_type,
            )
            self._local_regs.append(reg)

        slots = [
            (base_addr + i * slot_bytes, slot_bytes, dev_id) for i in range(num_slots)
        ]
        # NIXL_INIT_AGENT tags the local side; the object side is tagged with
        # this agent's own name (the store is local to it).
        self._local_handle = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", slots, mem_type
        )
        self._slot_bytes = slot_bytes

    def query_existence(self, obj_names: Iterable[str]) -> list[bool]:
        """One NIXL existence query; the store counts it as an access."""
        descriptors = [
            (_PROBE_ADDR, _PROBE_LEN, _PROBE_DEV_ID, name) for name in obj_names
        ]
        results = self._agent.query_memory(descriptors, self.backend, "OBJ")
        return [r is not None for r in results]

    def submit(
        self,
        job_id: int,
        slot_ids: np.ndarray,
        obj_names: Iterable[str],
        op: str,
    ) -> None:
        """Post a transfer between local slots and objects, pairwise.

        WRITE copies slots to objects, READ objects to slots. A transfer that
        cannot be posted is reported failed by the next ``poll``.
        """
        assert self._local_handle is not None, "register_local() first"
        # NIXL takes indices as int32; a wider dtype silently costs a conversion.
        assert slot_ids.dtype == np.int32
        # The OBJ backend maps devId -> obj_key. All descriptors must have
        # unique devIds or later registrations overwrite earlier ones.
        nixl_files = [
            (0, self._slot_bytes, dev_id, name)
            for dev_id, name in enumerate(obj_names, self._next_obj_dev_id)
        ]
        self._next_obj_dev_id += len(nixl_files)

        files_desc = self._agent.register_memory(nixl_files, "OBJ")
        if files_desc is None:
            logger.warning("register_memory (OBJ) failed for job %d", job_id)
            self._pending_results.append(ObjTransferResult(job_id, False))
            return

        obj_handle = self._agent.prep_xfer_dlist(self._agent_name, files_desc.trim())
        if not obj_handle:
            logger.warning("prep_xfer_dlist (OBJ) failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            self._pending_results.append(ObjTransferResult(job_id, False))
            return

        xfer_handle = self._agent.make_prepped_xfer(
            op,
            self._local_handle,
            slot_ids,
            obj_handle,
            np.arange(len(nixl_files), dtype=np.int32),
        )
        if not xfer_handle:
            logger.warning("make_prepped_xfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._pending_results.append(ObjTransferResult(job_id, False))
            return

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.warning("agent.transfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._agent.release_xfer_handle(xfer_handle)
            self._pending_results.append(ObjTransferResult(job_id, False))
            return

        self._transfers[job_id] = _Transfer(xfer_handle, files_desc, obj_handle)

    def _poll_transfers(self) -> None:
        """Poll every in-flight transfer once; move the ended ones (success or
        failure) into ``_pending_results`` and release their NIXL handles."""
        for job_id, entry in list(self._transfers.items()):
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                success = False
                logger.warning("check_xfer_state raised for job %d: %s", job_id, exc)
            else:
                if state == NIXL_PROC:
                    continue
                if state == NIXL_DONE:
                    success = True
                else:
                    success = False
                    logger.warning("transfer failed job=%d state=%s", job_id, state)

            transfer_time = None
            if success:
                # The duration only feeds metrics: losing it must not fail the
                # job or skip releasing the handle (e.g. NIXL_TELEMETRY_ENABLE=0
                # overrides the agent config and disables telemetry).
                try:
                    telemetry = self._agent.get_xfer_telemetry(entry.xfer_handle)
                    transfer_time = telemetry.xferDuration / 1e6
                except Exception as exc:
                    logger.debug(
                        "get_xfer_telemetry failed for job %d: %s", job_id, exc
                    )

            try:
                self._agent.release_xfer_handle(entry.xfer_handle)
            except Exception as exc:
                # Keep the entry until NIXL confirms that the transfer handle
                # can be released. The transfer may still access local memory,
                # so publishing its result would allow unsafe reuse.
                logger.warning("release_xfer_handle failed for job %d: %s", job_id, exc)
                continue

            # Once the transfer handle is released, these remaining cleanup
            # failures must not suppress the job completion. They can leak
            # NIXL metadata, but cannot leave an active data transfer behind.
            try:
                self._agent.release_dlist_handle(entry.obj_handle)
            except Exception as exc:
                logger.warning(
                    "release_dlist_handle failed for job %d: %s", job_id, exc
                )
            try:
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)

            del self._transfers[job_id]
            self._pending_results.append(
                ObjTransferResult(job_id, success, transfer_time)
            )

    def poll(self) -> list[ObjTransferResult]:
        """Return the transfers that ended since the last call."""
        self._poll_transfers()
        results, self._pending_results = self._pending_results, []
        return results

    def drain(self) -> None:
        """Block until every posted transfer has ended.

        NIXL exposes only ``check_xfer_state`` (poll-based), so this loops
        until no transfer is in flight. Results are returned by the next
        ``poll``.
        """
        start = time.monotonic()
        warned = False
        while self._transfers:
            self._poll_transfers()
            if not self._transfers:
                break
            if not warned and time.monotonic() - start > 5.0:
                logger.warning(
                    "ObjStoreClient.drain: still draining after 5s (%d transfers "
                    "in flight); a stuck transfer will block the engine.",
                    len(self._transfers),
                )
                warned = True
            time.sleep(0.001)

    def shutdown(self) -> None:
        """Release every NIXL handle and registration, in-flight ones included."""
        for job_id, entry in self._transfers.items():
            try:
                self._agent.release_xfer_handle(entry.xfer_handle)
            except Exception as exc:
                logger.warning("release_xfer_handle failed for job %d: %s", job_id, exc)
            try:
                self._agent.release_dlist_handle(entry.obj_handle)
            except Exception as exc:
                logger.warning(
                    "release_dlist_handle failed for job %d: %s", job_id, exc
                )
            try:
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)
        self._transfers.clear()
        if self._local_handle is not None:
            try:
                self._agent.release_dlist_handle(self._local_handle)
            except Exception as exc:
                logger.warning("failed to release local prepped handle: %s", exc)
            self._local_handle = None
        for reg in self._local_regs:
            try:
                self._agent.deregister_memory(reg)
            except Exception as exc:
                logger.warning("failed to deregister local buffer: %s", exc)
        self._local_regs = []
