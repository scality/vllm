# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tell a P/D proxy that the decode instance can start loading a prompt.

A prefill request may carry kv_transfer_params.notify_stored = {"url": str,
"tokens_left": int, "tokens_left_ratio": float}: once the store holds all of
the prompt a decode instance's wait_for_stored waits for but the larger of
tokens_left and that share of the prompt (and at least one chunk), the
scheduler POSTs {"request_id", "stored_tokens", "num_prompt_tokens"} to the
URL, once. The decode request can then load what is stored while the prefill
instance computes the rest, instead of after it answers.

The URL comes from the request, so only hosts the connector configuration
lists (notify_stored_hosts) are notified. POSTs go out from a background
thread: a slow or unreachable proxy delays nothing but its notifications.
"""

import json
import queue
import threading
import urllib.request
from collections.abc import Collection
from urllib.parse import urlsplit

from vllm.logger import init_logger

logger = init_logger(__name__)


def _endpoint(url: str) -> tuple[str, str] | None:
    """(host, host:port) of an http(s) URL, or None."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
    return parts.hostname, f"{parts.hostname}:{port}"


class StoredNotifier:
    def __init__(self, allowed_hosts: Collection[str], timeout_s: float = 2.0):
        """Args:
        allowed_hosts: Hosts that may be notified, as "host" (any port) or
            "host:port".
        timeout_s: Timeout of one POST.

        """
        self._allowed = frozenset(allowed_hosts)
        self._timeout_s = timeout_s
        self._queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def allows(self, url: str) -> bool:
        endpoint = _endpoint(url)
        return endpoint is not None and not self._allowed.isdisjoint(endpoint)

    def notify(self, url: str, payload: dict) -> bool:
        """Queue a POST of payload to url; False if url is not allowed."""
        if not self.allows(url):
            logger.warning_once(
                "notify_stored URL %s is not in notify_stored_hosts; not notifying",
                url,
            )
            return False
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="notify-stored", daemon=True
            )
            self._thread.start()
        self._queue.put((url, payload))
        return True

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            url, payload = item
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s):
                    pass
            except Exception as e:
                logger.warning("notify_stored POST to %s failed: %s", url, e)

    def shutdown(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=self._timeout_s + 1)
            self._thread = None
