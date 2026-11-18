# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.kv_offload.obj.client import ObjStoreClient

pytestmark = pytest.mark.cpu_test


def test_existence_queries_never_overlap():
    """Lookups and reuse notifications query from their own threads; one NIXL
    agent fails or hangs when queried concurrently."""
    inside = 0
    most_inside = 0
    guard = threading.Lock()

    def query_memory(descriptors, backend, mem_type):
        nonlocal inside, most_inside
        with guard:
            inside += 1
            most_inside = max(most_inside, inside)
        time.sleep(0.001)
        with guard:
            inside -= 1
        return [None] * len(descriptors)

    agent = MagicMock()
    agent.get_plugin_mem_types.return_value = ["DRAM_SEG", "OBJ_SEG"]
    agent.query_memory.side_effect = query_memory
    with (
        patch("vllm.v1.kv_offload.obj.client.nixl_agent_config"),
        patch("vllm.v1.kv_offload.obj.client.nixl_agent", return_value=agent),
    ):
        client = ObjStoreClient(
            "test", {"backend": "REST", "endpoint_override": "mock:9000"}, 1
        )
    agent.query_memory.reset_mock()

    threads = [
        threading.Thread(
            target=lambda: [client.query_existence(["a", "b"]) for _ in range(20)]
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert agent.query_memory.call_count == 80
    assert most_inside == 1
