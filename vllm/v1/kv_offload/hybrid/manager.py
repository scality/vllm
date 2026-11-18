# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of offloading KV to host memory and an object store at once.

Every chunk a request computes is stored to both. A load reads the chunks host
memory holds from there and the others from the object store; when both hold
enough of them, the load is split so that both finish together. Chunks a
request loaded or found in the GPU prefix cache are stored to host memory if
it lacks them, so that later loads on this server can be split.
"""

from collections.abc import Collection, Iterable

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager
from vllm.v1.metrics.cache_hit_source import CacheHitSource


def split_load(in_cpu: list[bool], in_obj: list[bool], obj_share: float) -> list[bool]:
    """Which chunks of a load to read from the object store.

    Chunks host memory lacks come from the object store. Of those both hold,
    the last ones go to the object store too, until it reads obj_share of the
    chunks, or as close as the others allow.
    """
    from_obj = [not cpu for cpu in in_cpu]
    both = [i for i, (cpu, obj) in enumerate(zip(in_cpu, in_obj)) if cpu and obj]
    extra = round(len(in_cpu) * obj_share) - sum(from_obj)
    for i in both[len(both) - min(max(extra, 0), len(both)) :]:
        from_obj[i] = True
    return from_obj


class HybridOffloadingManager(OffloadingManager):
    def __init__(
        self,
        cpu: CPUOffloadingManager,
        obj: ObjOffloadingManager,
        obj_load_share: float,
    ):
        """Args:
        cpu: Tracks the chunks in host memory.
        obj: Tracks the chunks in the object store.
        obj_load_share: Fraction of the chunks of a load read from the
            object store when host memory holds them too.

        """
        self._cpu = cpu
        self._obj = obj
        self._obj_load_share = obj_load_share
        # Chunks of a request's load read from host memory: a request loads
        # once at a time.
        self._cpu_loads: dict[str, list[OffloadKey]] = {}
        self._obj_loads: dict[str, list[OffloadKey]] = {}
        # Chunks each request loaded, which go back to host memory only.
        self._loaded: dict[str, set[OffloadKey]] = {}
        self._cpu_storing: set[OffloadKey] = set()
        self._obj_storing: set[OffloadKey] = set()

    def _in_cpu(self, key: OffloadKey, req_context: ReqContext) -> bool:
        return self._cpu.lookup(key, req_context) is LookupResult.HIT

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._cpu.lookup(key, req_context)
        if result is not LookupResult.MISS:
            return result
        return self._obj.lookup(key, req_context)

    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        if self._in_cpu(key, req_context):
            return CacheHitSource.HOST
        return self._obj.get_load_source(key, req_context)

    def requery_misses(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        self._obj.requery_misses(keys, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> HybridLoadStoreSpec:
        keys = list(keys)
        from_obj = split_load(
            [self._in_cpu(key, req_context) for key in keys],
            [self._obj.holds(key) for key in keys],
            self._obj_load_share,
        )
        cpu_chunks = [i for i, obj in enumerate(from_obj) if not obj]
        obj_chunks = [i for i, obj in enumerate(from_obj) if obj]
        cpu_keys = [keys[i] for i in cpu_chunks]
        obj_keys = [keys[i] for i in obj_chunks]
        req_id = req_context.req_id
        self._loaded.setdefault(req_id, set()).update(keys)
        cpu_spec = CPULoadStoreSpec([])
        if cpu_keys:
            spec = self._cpu.prepare_load(cpu_keys, req_context)
            assert isinstance(spec, CPULoadStoreSpec)
            cpu_spec = spec
            self._cpu_loads[req_id] = cpu_keys
        if obj_keys:
            self._obj_loads[req_id] = obj_keys
        moved = [key for key in obj_keys if self._in_cpu(key, req_context)]
        if moved:
            self._cpu.touch(moved, req_context)
        obj_spec = (
            self._obj.prepare_load(obj_keys, req_context)
            if obj_keys
            else ObjLoadStoreSpec([])
        )
        return HybridLoadStoreSpec(cpu_chunks, cpu_spec, obj_chunks, obj_spec)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._cpu.touch(keys, req_context)

    def record_access(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        self._cpu.record_access(keys, req_context)
        self._obj.record_access(keys, req_context)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        cpu_keys = self._cpu_loads.pop(req_context.req_id, None)
        if cpu_keys:
            self._cpu.complete_load(cpu_keys, req_context)
        obj_keys = self._obj_loads.pop(req_context.req_id, None)
        if obj_keys:
            # Undone by mark_load_failed() if the load failed.
            self._obj.note_loaded(obj_keys)

    def mark_load_failed(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        # Host memory does not fail loads; the chunks read from it are then
        # no longer split, which costs nothing.
        self._obj.mark_load_failed(keys, req_context)
        loaded = self._loaded.get(req_context.req_id)
        if loaded is not None:
            loaded.difference_update(keys)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        keys = list(keys)
        loaded = self._loaded.get(req_context.req_id, ())
        obj_out = self._obj.prepare_store(
            [key for key in keys if key not in loaded], req_context
        )
        if obj_out is None:
            return None
        cpu_out = self._cpu.prepare_store(keys, req_context)
        cpu_rows = (
            {key: row for row, key in enumerate(cpu_out.keys_to_store)}
            if cpu_out is not None
            else {}
        )
        obj_rows = {key: row for row, key in enumerate(obj_out.keys_to_store)}
        keys_to_store = [key for key in keys if key in cpu_rows or key in obj_rows]
        cpu_chunks = [i for i, key in enumerate(keys_to_store) if key in cpu_rows]
        obj_chunks = [i for i, key in enumerate(keys_to_store) if key in obj_rows]
        cpu_spec = CPULoadStoreSpec([])
        if cpu_out is not None:
            assert isinstance(cpu_out.store_spec, CPULoadStoreSpec)
            cpu_spec = cpu_out.store_spec.select(
                [cpu_rows[keys_to_store[i]] for i in cpu_chunks]
            )
        assert isinstance(obj_out.store_spec, ObjLoadStoreSpec)
        obj_spec = obj_out.store_spec.select(
            [obj_rows[keys_to_store[i]] for i in obj_chunks]
        )
        self._cpu_storing.update(cpu_rows)
        self._obj_storing.update(obj_rows)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=HybridLoadStoreSpec(cpu_chunks, cpu_spec, obj_chunks, obj_spec),
            evicted_keys=cpu_out.evicted_keys if cpu_out is not None else [],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        cpu_keys = [key for key in keys if key in self._cpu_storing]
        obj_keys = [key for key in keys if key in self._obj_storing]
        self._cpu_storing.difference_update(cpu_keys)
        self._obj_storing.difference_update(obj_keys)
        if cpu_keys:
            self._cpu.complete_store(cpu_keys, req_context, success)
        if obj_keys:
            self._obj.complete_store(obj_keys, req_context, success)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._obj.on_new_request(req_context)
        self._cpu.on_new_request(req_context)
        # Offers the chunks the request did not compute for storing too.
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._loaded.pop(req_context.req_id, None)
        self._cpu.on_request_finished(req_context)
        self._obj.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._cpu.take_events()

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._cpu.on_schedule_end(context)
        self._obj.on_schedule_end(context)

    def has_pending_work(self) -> bool:
        return self._cpu.has_pending_work() or self._obj.has_pending_work()

    def reset_cache(self) -> None:
        self._cpu.reset_cache()
        self._obj.reset_cache()
        self._cpu_loads.clear()
        self._obj_loads.clear()
        self._loaded.clear()
        self._cpu_storing.clear()
        self._obj_storing.clear()

    def get_stats(self):
        return self._cpu.get_stats()

    def shutdown(self) -> None:
        self._cpu.shutdown()
        self._obj.shutdown()
