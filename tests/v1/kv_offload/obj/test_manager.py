# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ObjOffloadingManager against an in-memory object store."""

import time
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.obj.common import ObjLoadStoreSpec
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager

pytestmark = pytest.mark.cpu_test

_STEP = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


def _key(i: int) -> OffloadKey:
    return make_offload_key(i.to_bytes(8, "big"), 0)


def _name(key: OffloadKey) -> str:
    return key.hex()


@pytest.fixture
def store() -> set[str]:
    return set()


@pytest.fixture
def manager(store: set[str]):
    file_mapper = MagicMock()
    file_mapper.get_file_name.side_effect = _name
    client = MagicMock()
    client.query_existence.side_effect = lambda names: [n in store for n in names]
    manager = ObjOffloadingManager(
        file_mapper=file_mapper, client=client, max_inflight_store_chunks=4
    )
    yield manager
    manager.shutdown()


def _lookup(manager: ObjOffloadingManager, key: OffloadKey, ctx: ReqContext):
    """Look a key up until the existence query has answered."""
    deadline = time.monotonic() + 5
    while (result := manager.lookup(key, ctx)) is LookupResult.RETRY:
        assert time.monotonic() < deadline
        manager.on_schedule_end(_STEP)
        time.sleep(0.01)
    return result


def test_lookup_queries_the_store(manager, store):
    store.add(_name(_key(1)))
    ctx = ReqContext(req_id="r")

    assert _lookup(manager, _key(1), ctx) is LookupResult.HIT
    assert _lookup(manager, _key(2), ctx) is LookupResult.MISS


def test_chunks_stored_here_need_no_query(manager):
    ctx = ReqContext(req_id="r")
    out = manager.prepare_store([_key(1)], ctx)
    assert out is not None

    assert manager.lookup(_key(1), ctx) is LookupResult.HIT_PENDING
    manager.complete_store(out.keys_to_store, ctx)
    assert manager.lookup(_key(1), ctx) is LookupResult.HIT


def test_failed_store_is_not_found(manager):
    ctx = ReqContext(req_id="r")
    manager.prepare_store([_key(1)], ctx)
    manager.complete_store([_key(1)], ctx, success=False)

    assert _lookup(manager, _key(1), ctx) is LookupResult.MISS


def test_prepare_store_skips_chunks_stored_or_being_stored(manager):
    ctx = ReqContext(req_id="r")
    manager.prepare_store([_key(1)], ctx)
    manager.complete_store([_key(1)], ctx)
    manager.prepare_store([_key(2)], ctx)

    out = manager.prepare_store([_key(1), _key(2), _key(3)], ctx)

    assert out is not None
    assert out.keys_to_store == [_key(3)]
    assert isinstance(out.store_spec, ObjLoadStoreSpec)
    assert out.store_spec.obj_names == [_name(_key(3))]


def test_prepare_store_declines_past_the_in_flight_limit(manager):
    ctx = ReqContext(req_id="r")
    assert manager.prepare_store([_key(i) for i in range(3)], ctx) is not None

    assert manager.prepare_store([_key(3), _key(4)], ctx) is None
    assert manager.prepare_store([_key(3)], ctx) is not None


def test_failed_load_is_a_miss_for_the_request(manager, store):
    """A chunk whose load failed (evicted since it was found) is not offered
    again to the request, which recomputes it."""
    store.add(_name(_key(1)))
    ctx = ReqContext(req_id="r")
    assert _lookup(manager, _key(1), ctx) is LookupResult.HIT
    spec = manager.prepare_load([_key(1)], ctx)
    assert spec.obj_names == [_name(_key(1))]

    manager.complete_load([_key(1)], ctx)
    manager.mark_load_failed([_key(1)], ctx)

    assert manager.lookup(_key(1), ctx) is LookupResult.MISS
