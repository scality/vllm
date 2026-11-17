# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of offloading KV straight from the GPU to an object store."""

from collections import OrderedDict
from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.obj.client import ObjStoreClient
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.obj.lookup import ObjAsyncLookupManager, ReuseNotifier

logger = init_logger(__name__)


class ObjOffloadingManager(OffloadingManager):
    """Tracks which chunks are in the object store; it holds no data.

    A chunk is found by an existence query, batched per scheduler step, or
    because this instance stored it. Nothing is evicted here: the store
    evicts, and a load of an evicted chunk fails and is recomputed.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        client: ObjStoreClient,
        max_inflight_store_chunks: int,
        max_stored_keys: int = 100_000,
        reuse_notify: dict | None = None,
    ):
        """Args:
        file_mapper: Maps a key to its object name.
        client: Used only for existence queries.
        max_inflight_store_chunks: Chunks being stored above which
            prepare_store() declines; the scheduler retries next step.
        max_stored_keys: Chunks this instance stored that it remembers, so
            their lookups need no query.
        reuse_notify: See ReuseNotifier.from_config.

        """
        self._file_mapper = file_mapper
        self._client = client
        self._max_inflight_store_chunks = max_inflight_store_chunks
        self._max_stored_keys = max_stored_keys
        self._storing: set[OffloadKey] = set()
        self._stored: OrderedDict[OffloadKey, None] = OrderedDict()
        self._lookup_manager = ObjAsyncLookupManager(
            self._query_existence, tier_type="obj"
        )
        self._reuse_notifier = ReuseNotifier.from_config(
            self._query_existence, reuse_notify
        )

    def _query_existence(self, keys: Iterable[OffloadKey]) -> list[bool]:
        return self._client.query_existence(
            self._file_mapper.get_file_name(k) for k in keys
        )

    def _obj_names(self, keys: Iterable[OffloadKey]) -> list[str]:
        return [self._file_mapper.get_file_name(k) for k in keys]

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._storing:
            return LookupResult.HIT_PENDING
        if key in self._stored:
            self._stored.move_to_end(key)
            return LookupResult.HIT
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def requery_misses(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        self._lookup_manager.requery(keys, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> ObjLoadStoreSpec:
        if self._reuse_notifier is not None:
            self._reuse_notifier.note_loaded(keys)
        return ObjLoadStoreSpec(self._obj_names(keys))

    def mark_load_failed(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            self._stored.pop(key, None)
        self._lookup_manager.mark_miss(keys)

    def record_access(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        if self._reuse_notifier is not None:
            self._reuse_notifier.add(keys)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        keys_to_store = [
            k for k in keys if k not in self._storing and k not in self._stored
        ]
        if not keys_to_store:
            return PrepareStoreOutput([], ObjLoadStoreSpec([]), [])
        if len(self._storing) + len(keys_to_store) > self._max_inflight_store_chunks:
            return None
        self._storing.update(keys_to_store)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=ObjLoadStoreSpec(self._obj_names(keys_to_store)),
            evicted_keys=[],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        self._storing.difference_update(keys)
        if not success:
            return
        for key in keys:
            self._stored[key] = None
            self._stored.move_to_end(key)
        while len(self._stored) > self._max_stored_keys:
            self._stored.popitem(last=False)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    def reset_cache(self) -> None:
        self._stored.clear()

    def shutdown(self) -> None:
        if self._reuse_notifier is not None:
            self._reuse_notifier.stop()
        self._lookup_manager.shutdown()
        self._client.shutdown()
