# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.notify_stored import (
    StoredNotifier,
)

pytestmark = pytest.mark.cpu_test


def test_only_listed_hosts_are_notified():
    notifier = StoredNotifier(["proxy", "other:8100"])
    assert notifier.allows("http://proxy/ready/1")
    assert notifier.allows("http://proxy:9000/ready/1")
    assert notifier.allows("http://other:8100/ready/1")
    assert not notifier.allows("http://other:8101/ready/1")
    assert not notifier.allows("http://elsewhere/ready/1")
    assert not notifier.allows("file:///etc/passwd")
    assert not notifier.notify("http://elsewhere/ready/1", {})


def test_notify_posts_the_payload():
    received: list[tuple[str, dict]] = []
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, json.loads(body)))
            self.send_response(204)
            self.end_headers()
            done.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    notifier = StoredNotifier(["127.0.0.1"])
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/ready/7"
        assert notifier.notify(url, {"stored_tokens": 12})
        assert done.wait(5)
        assert received == [("/ready/7", {"stored_tokens": 12})]
    finally:
        notifier.shutdown()
        server.shutdown()
