# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload KV to host memory and, straight from the GPU, to an object store.

Loads read from both at once: host memory over each GPU's PCIe link, the
object store over the storage NICs. kv_connector_extra_config holds the
options of CPUOffloadingSpec and ObjOffloadingSpec, and:

    spec_name: "HybridOffloadingSpec"
    obj_load_share: fraction of the chunks of a load read from the object
        store when host memory holds them too (default 0.5). Best is
        R / (R + H), with R the rate at which the object store delivers a
        chunk to all workers, H the rate at which host memory does.

With more than one worker, set canonical_layout: every worker then stores,
as the object store needs.
"""

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.hybrid.manager import HybridOffloadingManager
from vllm.v1.kv_offload.obj.manager import ObjOffloadingManager
from vllm.v1.kv_offload.obj.spec import ObjOffloadingSpec


class HybridOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        if config.parallel.world_size > 1 and not config.canonical_layout:
            raise ValueError(
                "HybridOffloadingSpec needs canonical_layout with several workers"
            )
        self.canonical_layout = config.canonical_layout
        self._obj = ObjOffloadingSpec(config)
        self.obj_load_share = float(self.extra_config.get("obj_load_share", 0.5))
        if not 0.0 <= self.obj_load_share <= 1.0:
            raise ValueError(
                f"obj_load_share must be in [0, 1], got {self.obj_load_share}"
            )
        self._hybrid_manager: HybridOffloadingManager | None = None
        self._hybrid_worker: OffloadingWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._hybrid_manager is None:
            cpu = super().get_manager()
            obj = self._obj.get_manager()
            assert isinstance(cpu, CPUOffloadingManager)
            assert isinstance(obj, ObjOffloadingManager)
            self._hybrid_manager = HybridOffloadingManager(
                cpu, obj, self.obj_load_share
            )
        return self._hybrid_manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._hybrid_worker is None:
            from vllm.v1.kv_offload.hybrid.worker import HybridOffloadingWorker

            obj = self._obj.get_worker(kv_caches)
            self._hybrid_worker = HybridOffloadingWorker(
                super().get_worker(kv_caches), obj, self.blocks_per_chunk
            )
        return self._hybrid_worker
