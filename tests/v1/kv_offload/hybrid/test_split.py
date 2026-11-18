# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.hybrid.split import LoadTimeModel, shared_chunks_from_obj

pytestmark = pytest.mark.cpu_test


def _fed(
    model: LoadTimeModel, latency: float, per_chunk: float, alone: bool = True
) -> LoadTimeModel:
    for i in range(30):
        for n in (8, 64, 512):
            model.observe(n, latency + n * per_chunk, now=float(i), alone=alone)
    return model


def test_model_learns_latency_and_rate_from_loads():
    model = _fed(LoadTimeModel(latency=0.05, per_chunk=0.005), 0.002, 0.0001)
    assert model.latency == pytest.approx(0.002, abs=0.002)
    assert model.per_chunk == pytest.approx(0.0001, rel=0.1)
    assert model.contention == 1.0


def test_shared_loads_only_fit_the_contention():
    model = _fed(LoadTimeModel(latency=0.05, per_chunk=0.005), 0.002, 0.0001)
    _fed(model, 0.004, 0.0002, alone=False)
    assert model.per_chunk == pytest.approx(0.0001, rel=0.1)
    assert model.contention == pytest.approx(2.0, rel=0.1)
    assert model.shared().seconds(100) == pytest.approx(0.024, rel=0.15)


def test_model_returns_to_its_prior_without_loads():
    model = _fed(LoadTimeModel(latency=0.05, per_chunk=0.005), 0.002, 0.0001)
    _fed(model, 0.004, 0.0002, alone=False)
    model.refresh(now=36000.0)
    assert model.latency == pytest.approx(0.05, rel=0.05)
    assert model.per_chunk == pytest.approx(0.005, rel=0.05)
    assert model.contention == pytest.approx(1.0, rel=0.05)


def test_short_loads_stay_in_host_memory():
    cpu = LoadTimeModel(latency=0.001, per_chunk=0.001)
    obj = LoadTimeModel(latency=0.010, per_chunk=0.001)
    assert shared_chunks_from_obj(0, 0, 4, cpu, obj) == 0


def test_long_loads_end_together():
    cpu = LoadTimeModel(latency=0.001, per_chunk=0.001)
    obj = LoadTimeModel(latency=0.010, per_chunk=0.001)
    # 1 ms + (100 - x) ms == 10 ms + x ms
    assert shared_chunks_from_obj(0, 0, 100, cpu, obj) in (45, 46)
    # Chunks only one medium holds load from it and shift the balance.
    assert shared_chunks_from_obj(0, 20, 100, cpu, obj) in (35, 36)
    slow_obj = LoadTimeModel(latency=0.010, per_chunk=0.003)
    assert shared_chunks_from_obj(0, 0, 100, cpu, slow_obj) in (22, 23)


def test_contention_can_make_splitting_slower_than_host_memory_alone():
    cpu = LoadTimeModel(latency=0.001, per_chunk=0.001)
    obj = LoadTimeModel(latency=0.010, per_chunk=0.001)
    for i in range(20):
        # Both media at a third of their speed when they load together.
        cpu.observe(50, 3 * (0.001 + 50 * 0.001), now=float(i), alone=False)
        obj.observe(50, 3 * (0.010 + 50 * 0.001), now=float(i), alone=False)
    assert shared_chunks_from_obj(0, 0, 100, cpu, obj) == 0
