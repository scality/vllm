# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

import ctypes
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, ClassVar

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.obj.client import NIXL_READ, NIXL_WRITE, ObjStoreClient
from vllm.v1.kv_offload.obj.lookup import ObjAsyncLookupManager, ReuseNotifier
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class ObjectStoreSecondaryTierManager(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier. Object keys are formed as ``{prefix}/{hash_shard}/{hash}.bin``.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        store_config: dict,
        prefix: str = "",
        io_threads: int = 4,
        enable_kv_events: bool = False,
        locality: str | None = None,
        backpressure_detector: BackpressureDetector | None = None,
        reuse_notify: dict | None = None,
    ):
        """Args:
        offloading_spec: Offloading configuration.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        store_config: Object store connection parameters (see ObjStoreConfig).
        prefix: Key prefix prepended to all object keys.
        io_threads: Number of NIXL I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.
        reuse_notify: Send the keys reused from the GPU or the primary tier
            to the store, as existence queries. ``{"enabled": bool,
            "interval_s": float, "max_keys": int}``, off by default. Keep
            ``interval_s`` at most half the shortest time a block stays in the
            store's eviction queue, so a reused block is reported before it
            can be evicted.

        """
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            backpressure_detector,
        )
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed download can
        # mark its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}

        root_dir = f"{prefix}/" if prefix else ""
        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self._file_mapper = FileMapper.from_offloading_spec(
            root_dir, offloading_spec, parallel_agnostic=True
        )

        self._client = ObjStoreClient("ObjAgent", store_config, io_threads)
        try:
            self._client.register_local(
                base_addr=ctypes.addressof(ctypes.c_char.from_buffer(primary_kv_view)),
                num_slots=len(primary_kv_view),
                slot_bytes=self.block_size_bytes,
            )
        except Exception:
            self._client.shutdown()
            raise

        self._lookup_manager = ObjAsyncLookupManager(
            self._query_existence, tier_type=self.tier_type
        )

        self._reuse_notifier = ReuseNotifier.from_config(
            self._query_existence, reuse_notify
        )

    def _query_existence(self, keys: Iterable[OffloadKey]) -> list[bool]:
        """One existence query for ``keys``."""
        return self._client.query_existence(
            self._file_mapper.get_file_name(k) for k in keys
        )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._client.submit(
            job_metadata.job_id, job_metadata.chunk_ids, obj_keys, NIXL_WRITE
        )

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._load_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        if self._reuse_notifier is not None:
            self._reuse_notifier.note_loaded(job_metadata.keys)
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._client.submit(
            job_metadata.job_id, job_metadata.chunk_ids, obj_keys, NIXL_READ
        )

    def requery_misses(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        self._lookup_manager.requery(keys, req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    def record_access(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        if self._reuse_notifier is not None:
            self._reuse_notifier.add(keys)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll transfers; a failed promotion marks its cached verdicts False
        here (scheduler thread)."""
        results = [
            JobResult(job_id=r.job_id, success=r.success, transfer_time=r.transfer_time)
            for r in self._client.poll()
        ]
        for result in results:
            if self.events is not None:
                keys = self._store_job_keys.pop(result.job_id, None)
                if result.success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            # Mark only the keys that did not load as a miss; the request
            # recomputes them. The miss is per-request (cleared when the request
            # finishes), so other requests still HIT the blocks that loaded
            # fine. nixl reports the batch as a whole (successful_keys is None),
            # so today this marks all keys; the subtraction keeps it correct if
            # partial results are ever reported.
            load_keys = self._load_job_keys.pop(result.job_id, None)
            if load_keys is not None and not result.success:
                successful = set(result.successful_keys or ())
                failed = [k for k in load_keys if k not in successful]
                self._lookup_manager.mark_miss(failed)
        return results

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def drain_jobs(self) -> None:
        """Block until every submitted transfer has completed or failed; their
        results are surfaced by the next ``get_finished_jobs()`` call."""
        self._client.drain()

    def shutdown(self) -> None:
        if self._reuse_notifier is not None:
            self._reuse_notifier.stop()
        self._lookup_manager.shutdown()
        self._client.shutdown()
