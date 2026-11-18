# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How a hybrid load splits the chunks both media hold.

Each medium's load time is modelled as latency + chunks * per_chunk, fitted
to the loads it made alone, times a contention factor fitted to the loads it
shared with the other medium: both may share PCIe paths. Nothing is probed.
Each load then sends to the object store the number of shared chunks that
makes both parts end together, or leaves the load to one medium when that is
sooner, as for short loads when the object store's latency outweighs what
splitting saves.
"""

import math
from typing import NamedTuple


class LinearTime(NamedTuple):
    latency: float
    per_chunk: float

    def seconds(self, chunks: int) -> float:
        return self.latency + chunks * self.per_chunk if chunks else 0.0


class LoadTimeModel:
    """latency + chunks * per_chunk seconds, fitted to (chunks, seconds)
    samples whose weight halves every half_life seconds.

    Latency and rate are told apart by a least squares fit as far as the
    samples' sizes vary; samples of one size only scale the prior to their
    mean time. Loads shared with the other medium only fit the contention
    factor, their time over the fit's. The prior, and a factor of 1, also
    count as prior_weight samples, so that with no recent loads the model
    returns to them.
    """

    def __init__(
        self,
        latency: float,
        per_chunk: float,
        half_life: float = 300.0,
        prior_weight: float = 0.02,
    ):
        assert latency >= 0 and per_chunk > 0 and half_life > 0
        self._prior = (latency, per_chunk)
        self._half_life = half_life
        self._prior_weight = prior_weight
        # Decayed weighted sums of 1, n, n^2, t, n*t over the loads alone,
        # and of 1, t / fitted t over the shared ones.
        self._sums = [0.0] * 5
        self._shared_sums = [0.0] * 2
        self._time: float | None = None
        self.latency = latency
        self.per_chunk = per_chunk
        self.contention = 1.0

    def observe(self, chunks: int, seconds: float, now: float, alone: bool) -> None:
        """A load of this many chunks took this long, at time now, alone or
        while the other medium loaded too."""
        if chunks <= 0 or not math.isfinite(seconds) or seconds < 0:
            return
        self._decay(now)
        if alone:
            for i, value in enumerate(
                (1.0, chunks, chunks * chunks, seconds, chunks * seconds)
            ):
                self._sums[i] += value
            self._fit()
        else:
            self._shared_sums[0] += 1.0
            self._shared_sums[1] += seconds / self.alone().seconds(chunks)
        self._fit_contention()

    def refresh(self, now: float) -> None:
        """Let the samples age, towards the prior, up to time now."""
        self._decay(now)
        self._fit()
        self._fit_contention()

    def alone(self) -> LinearTime:
        return LinearTime(self.latency, self.per_chunk)

    def shared(self) -> LinearTime:
        return LinearTime(
            self.latency * self.contention, self.per_chunk * self.contention
        )

    def _decay(self, now: float) -> None:
        if self._time is not None and now > self._time:
            factor = 0.5 ** ((now - self._time) / self._half_life)
            self._sums = [s * factor for s in self._sums]
            self._shared_sums = [s * factor for s in self._shared_sums]
        self._time = now if self._time is None else max(self._time, now)

    def _fit_contention(self) -> None:
        w, ratios = self._shared_sums
        self.contention = (ratios + self._prior_weight) / (w + self._prior_weight)

    def _fit(self) -> None:
        w, n, nn, t, nt = self._sums
        prior_latency, prior_per_chunk = self._prior
        if w <= 0 or n <= 0:
            self.latency, self.per_chunk = self._prior
            return
        mean_n, mean_t = n / w, t / w
        # The prior's shape, scaled to the samples' mean time.
        scale = mean_t / (prior_latency + prior_per_chunk * mean_n)
        latency, per_chunk = scale * prior_latency, scale * prior_per_chunk
        # As far as the sizes vary, a least squares fit, if it makes sense.
        spread = max(nn / w - mean_n * mean_n, 0.0) / (mean_n * mean_n)
        fit_per_chunk = (nt / w - mean_n * mean_t) / (nn / w - mean_n * mean_n or 1)
        fit_latency = mean_t - fit_per_chunk * mean_n
        if spread > 0 and fit_per_chunk > 0 and fit_latency >= 0:
            trust = spread / (spread + 0.25)
            latency += trust * (fit_latency - latency)
            per_chunk += trust * (fit_per_chunk - per_chunk)
        prior = self._prior_weight / (self._prior_weight + w)
        self.latency = latency + prior * (prior_latency - latency)
        self.per_chunk = per_chunk + prior * (prior_per_chunk - per_chunk)


def shared_chunks_from_obj(
    cpu_only: int,
    obj_only: int,
    shared: int,
    cpu: LoadTimeModel,
    obj: LoadTimeModel,
) -> int:
    """How many of the shared chunks to load from the object store so that the
    load ends soonest: both parts end together, or one medium does it all."""
    cpu_shared, obj_shared = cpu.shared(), obj.shared()

    def seconds(x: int) -> float:
        cpu_chunks, obj_chunks = cpu_only + shared - x, obj_only + x
        if cpu_chunks and obj_chunks:
            return max(cpu_shared.seconds(cpu_chunks), obj_shared.seconds(obj_chunks))
        return cpu.alone().seconds(cpu_chunks) + obj.alone().seconds(obj_chunks)

    # cpu_shared.seconds(cpu_only + shared - x) == obj_shared.seconds(obj_only + x)
    balance = (
        cpu_shared.latency
        - obj_shared.latency
        + (cpu_only + shared) * cpu_shared.per_chunk
        - obj_only * obj_shared.per_chunk
    ) / (cpu_shared.per_chunk + obj_shared.per_chunk)
    candidates = {0, shared}
    if 0 < balance < shared:
        candidates |= {math.floor(balance), math.ceil(balance)}
    return min(sorted(candidates), key=seconds)
