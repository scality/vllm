# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

import ctypes
import threading
import time
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
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
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
    from collections.abc import Callable

    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class ObjAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for ObjectStoreSecondaryTierManager.

    Batches existence probes into a single query_memory() call so the
    background thread issues one round-trip per step instead of one per key.
    """

    def __init__(
        self,
        tier: "ObjectStoreSecondaryTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return self._tier._query_existence(keys)


class _ReuseNotifier:
    """Sends the keys reused above the obj tier to the object store.

    The scheduler thread only adds keys to a set; a background thread sends
    each distinct key once per interval as an existence query, which the store
    treats as an access. Nothing here waits on the store.
    """

    def __init__(
        self,
        send: "Callable[[list[OffloadKey]], object]",
        interval_s: float,
        max_keys: int,
    ) -> None:
        self._send = send
        self._interval_s = interval_s
        self._max_keys = max_keys
        self._lock = threading.Lock()
        # Keys reused from the GPU or the CPU tier, sent at the next flush.
        self._pending: set[OffloadKey] = set()
        # Keys this tier read from the store: the store saw that access already.
        self._loaded: set[OffloadKey] = set()
        self._dropped = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm_offloading_obj_reuse_notify", daemon=True
        )
        self._thread.start()

    def add(self, keys: Collection[OffloadKey]) -> None:
        with self._lock:
            room = self._max_keys - len(self._pending)
            for key in keys:
                if key in self._pending:
                    continue
                if room <= 0:
                    self._dropped += 1
                    continue
                self._pending.add(key)
                room -= 1
            full = room <= 0
        if full:
            self._wake.set()

    def note_loaded(self, keys: Collection[OffloadKey]) -> None:
        with self._lock:
            # Past the cap, a key is not filtered and costs one redundant HEAD.
            for key in keys:
                if len(self._loaded) >= self._max_keys:
                    break
                self._loaded.add(key)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=self._interval_s + 5)

    def _run(self) -> None:
        while True:
            self._wake.wait(self._interval_s)
            self._wake.clear()
            self._flush()
            if self._stop.is_set():
                return

    def _flush(self) -> None:
        with self._lock:
            keys, self._pending = self._pending, set()
            loaded, self._loaded = self._loaded, set()
            dropped, self._dropped = self._dropped, 0
        filtered = len(keys & loaded)
        keys -= loaded
        if not keys and not dropped and not filtered:
            return
        start = time.perf_counter()
        try:
            if keys:
                self._send(list(keys))
        except Exception as exc:
            logger.warning("OBJ reuse notify failed for %d keys: %s", len(keys), exc)
            return
        logger.debug(
            "OBJ reuse notify: %d keys sent in %.2f ms, %d filtered, %d dropped",
            len(keys),
            (time.perf_counter() - start) * 1000,
            filtered,
            dropped,
        )


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
            tier=self, tier_type=self.tier_type
        )

        reuse_notify = reuse_notify or {}
        self._reuse_notifier: _ReuseNotifier | None = None
        if reuse_notify.get("enabled", False):
            self._reuse_notifier = _ReuseNotifier(
                self._query_existence,
                interval_s=float(reuse_notify.get("interval_s", 5.0)),
                max_keys=int(reuse_notify.get("max_keys", 10000)),
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
