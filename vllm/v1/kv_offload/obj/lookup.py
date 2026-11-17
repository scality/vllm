# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Existence lookups and reuse notifications against the object store."""

import threading
import time
from collections.abc import Callable, Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager

logger = init_logger(__name__)

QueryExistence = Callable[[list[OffloadKey]], list[bool]]


class ObjAsyncLookupManager(AsyncLookupManager):
    """Batches existence probes into a single query_memory() call so the
    background thread issues one round-trip per step instead of one per key.
    """

    def __init__(self, query: QueryExistence, tier_type: str) -> None:
        super().__init__(tier_type=tier_type)
        self._query = query

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return self._query(keys)


class ReuseNotifier:
    """Sends the keys reused above the object store to the store.

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
        # Keys read from the store: the store saw that access already.
        self._loaded: set[OffloadKey] = set()
        self._dropped = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm_offloading_obj_reuse_notify", daemon=True
        )
        self._thread.start()

    @classmethod
    def from_config(
        cls, send: "Callable[[list[OffloadKey]], object]", config: dict | None
    ) -> "ReuseNotifier | None":
        """``{"enabled": bool, "interval_s": float, "max_keys": int}``, off by
        default."""
        config = config or {}
        if not config.get("enabled", False):
            return None
        return cls(
            send,
            interval_s=float(config.get("interval_s", 5.0)),
            max_keys=int(config.get("max_keys", 10000)),
        )

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
