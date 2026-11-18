# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HybridOffloadingManager over a CPU manager and an in-memory object store."""

from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.hybrid.manager import (
    HybridOffloadingManager,
    shared_from_obj_at_share,
    split_load,
)
from vllm.v1.kv_offload.hybrid.split import LoadTimeModel
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager

pytestmark = pytest.mark.cpu_test


def _key(i: int) -> OffloadKey:
    return make_offload_key(i.to_bytes(8, "big"), 0)


def _make_manager(num_cpu_chunks: int, obj_share: float | None = 0.5, clock=None):
    file_mapper = MagicMock()
    file_mapper.get_file_name.side_effect = lambda key: key.hex()
    client = MagicMock()
    client.query_existence.side_effect = lambda names: [False for _ in names]
    obj = ObjOffloadingManager(
        file_mapper=file_mapper, client=client, max_inflight_store_chunks=8
    )
    cpu = CPUOffloadingManager(num_chunks=num_cpu_chunks)
    # Host memory 1 ms + 1 ms per chunk, the object store 10 ms + 1 ms.
    return HybridOffloadingManager(
        cpu,
        obj,
        LoadTimeModel(latency=0.001, per_chunk=0.001),
        LoadTimeModel(latency=0.010, per_chunk=0.001),
        obj_share,
        **({"clock": clock} if clock else {}),
    )


@pytest.fixture
def make_manager():
    managers: list[HybridOffloadingManager] = []

    def make(**kwargs) -> HybridOffloadingManager:
        managers.append(_make_manager(**kwargs))
        return managers[-1]

    yield make
    for manager in managers:
        manager.shutdown()


@pytest.fixture
def ctx() -> ReqContext:
    return ReqContext(req_id="r")


def _store(manager: HybridOffloadingManager, keys, ctx) -> HybridLoadStoreSpec:
    out = manager.prepare_store(keys, ctx)
    assert out is not None
    manager.complete_store(out.keys_to_store, ctx)
    assert isinstance(out.store_spec, HybridLoadStoreSpec)
    return out.store_spec


@pytest.mark.parametrize(
    "in_cpu, in_obj, share, expected",
    [
        # Both hold all: the last half come from the object store.
        ([1, 1, 1, 1], [1, 1, 1, 1], 0.5, [0, 0, 1, 1]),
        # Chunks host memory lacks count towards the share.
        ([0, 1, 1, 1], [1, 1, 1, 1], 0.5, [1, 0, 0, 1]),
        # The object store lacks some: as close to the share as possible.
        ([1, 1, 1, 1], [0, 0, 0, 1], 0.75, [0, 0, 0, 1]),
        ([0, 0, 0, 1], [1, 1, 1, 1], 0.25, [1, 1, 1, 0]),
        ([1, 1], [1, 1], 0.0, [0, 0]),
        ([1, 1], [1, 1], 1.0, [1, 1]),
    ],
)
def test_split_load_at_a_share(in_cpu, in_obj, share, expected):
    in_cpu = [bool(c) for c in in_cpu]
    in_obj = [bool(o) for o in in_obj]
    from_obj = split_load(
        in_cpu, in_obj, shared_from_obj_at_share(in_cpu, in_obj, share)
    )
    assert from_obj == [bool(e) for e in expected]


def test_stores_go_to_both_and_loads_split(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=8)
    keys = [_key(i) for i in range(4)]
    spec = _store(manager, keys, ctx)
    assert spec.cpu_chunks == spec.obj_chunks == [0, 1, 2, 3]

    assert all(manager.lookup(k, ctx) is LookupResult.HIT for k in keys)
    load = manager.prepare_load(keys, ctx)
    assert load.cpu_chunks == [0, 1] and load.obj_chunks == [2, 3]
    assert load.obj.obj_names == [keys[2].hex(), keys[3].hex()]
    manager.complete_load(keys, ctx)


def test_chunks_evicted_from_host_memory_load_from_the_object_store(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=2, obj_share=0.0)
    old = [_key(0), _key(1)]
    _store(manager, old, ctx)
    spec = _store(manager, [_key(2), _key(3)], ctx)
    assert spec.cpu_chunks == [0, 1]  # evicted the old ones

    assert manager.lookup(_key(0), ctx) is LookupResult.HIT
    load = manager.prepare_load([_key(0), _key(2)], ctx)
    assert load.obj_chunks == [0] and load.cpu_chunks == [1]


def test_loads_pin_host_memory_until_complete(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=1, obj_share=0.0)
    _store(manager, [_key(0)], ctx)
    load = manager.prepare_load([_key(0)], ctx)
    assert load.cpu_chunks == [0]

    # Host memory is full of a chunk being loaded: the object store only.
    spec = _store(manager, [_key(1)], ReqContext(req_id="other"))
    assert spec.cpu_chunks == [] and spec.obj_chunks == [0]

    manager.complete_load([_key(0)], ctx)
    spec = _store(manager, [_key(2)], ReqContext(req_id="other"))
    assert spec.cpu_chunks == [0]


def test_failed_load_stops_splitting_those_chunks(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=4, obj_share=1.0)
    keys = [_key(0), _key(1)]
    _store(manager, keys, ctx)
    load = manager.prepare_load(keys, ctx)
    assert load.obj_chunks == [0, 1]
    manager.complete_load(keys, ctx)
    manager.mark_load_failed(keys, ctx)

    load = manager.prepare_load(keys, ctx)
    assert load.cpu_chunks == [0, 1] and load.obj_chunks == []


def test_object_store_backlog_holds_stores_back(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=16)
    keys = [_key(i) for i in range(9)]  # max_inflight_store_chunks is 8
    assert manager.prepare_store(keys, ctx) is None
    assert manager.lookup(keys[0], ctx) is not LookupResult.HIT


def test_loaded_chunks_go_back_to_host_memory_only(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=8)
    assert manager.on_new_request(ctx).policy is OffloadPolicy.REQUEST_LEVEL
    # Chunks another server stored: in the object store only.
    keys = [_key(0), _key(1)]
    load = manager.prepare_load(keys, ctx)
    assert load.obj_chunks == [0, 1]
    manager.complete_load(keys, ctx)

    spec = _store(manager, [*keys, _key(2)], ctx)
    assert spec.cpu_chunks == [0, 1, 2] and spec.obj_chunks == [2]

    # Another request finding them in the GPU prefix cache stores nothing.
    other = ReqContext(req_id="other")
    out = manager.prepare_store(keys, other)
    assert out is not None and out.keys_to_store == []


def test_chunks_of_a_failed_load_are_stored_to_both(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=8)
    keys = [_key(0)]
    manager.prepare_load(keys, ctx)
    manager.complete_load(keys, ctx)
    manager.mark_load_failed(keys, ctx)

    spec = _store(manager, keys, ctx)
    assert spec.cpu_chunks == [0] and spec.obj_chunks == [0]


def test_measured_load_times_move_the_split(make_manager, ctx):
    now = [0.0]
    manager = make_manager(num_cpu_chunks=64, obj_share=None, clock=lambda: now[0])
    keys = [_key(i) for i in range(40)]
    for start in range(0, 40, 8):
        _store(manager, keys[start : start + 8], ctx)

    # 1 ms + (40 - x) ms == 10 ms + x ms
    load = manager.prepare_load(keys, ctx)
    assert len(load.obj_chunks) in (15, 16)
    manager.complete_load(keys, ctx)

    # The object store turns out much slower than expected.
    for _ in range(20):
        now[0] += 1
        load = manager.prepare_load(keys, ctx)
        n_cpu, n_obj = len(load.cpu_chunks), len(load.obj_chunks)
        manager.record_load_times(
            keys, ctx, {"cpu": 0.001 + 0.001 * n_cpu, "obj": 0.050 + 0.004 * n_obj}
        )
        manager.complete_load(keys, ctx)
    load = manager.prepare_load(keys, ctx)
    assert len(load.obj_chunks) < 8
    manager.complete_load(keys, ctx)


def test_only_the_object_store_makes_chunks_shared(make_manager, ctx):
    manager = make_manager(num_cpu_chunks=8)
    out = manager.prepare_store([_key(0)], ctx)
    assert out is not None
    assert not manager.shared_holds(_key(0))  # still being stored
    manager.complete_store(out.keys_to_store, ctx)
    assert manager.shared_holds(_key(0))
