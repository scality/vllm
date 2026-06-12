# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from dataclasses import dataclass, field

# NIXL backend params that the accelerated engine owns or that the manager
# injects (``num_threads`` from ``io_threads``). They may not be overridden
# through ``extra_params``.
_RESERVED_ACCELERATED_PARAMS = frozenset(
    {"accelerated", "type", "endpoint_override", "num_threads"}
)

# S3-only params that are meaningless to an accelerated engine. Supplying any
# of them alongside ``accelerated`` indicates a mixed-mode config.
_S3_ONLY_FIELDS = ("bucket", "access_key", "secret_key")


def _is_truthy(value: object) -> bool:
    """Normalize a JSON bool or ``"true"``/``"false"`` string to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


@dataclass
class ObjStoreConfig:
    """Connection parameters for an object store backend.

    Two modes are supported:

    - **S3 mode** (default): drives NIXL's default AWS-SDK S3 engine over
      HTTP(S). Requires ``bucket``, ``endpoint_override``, ``access_key`` and
      ``secret_key``.
    - **Accelerated mode** (``accelerated=true``): selects a NIXL accelerated
      OBJ engine (e.g. ``scality_ai_connector``, ``dell``) that moves bytes
      over cuObject/GPUDirect RDMA. Requires ``type`` and ``endpoint_override``;
      ``bucket``/``access_key``/``secret_key``/``scheme`` are not used.
      Engine-specific params may be passed through ``extra_params``.
    """

    endpoint_override: str
    # Accelerated-engine params.
    accelerated: bool = False
    type: str = ""
    extra_params: dict[str, str] = field(default_factory=dict)
    # S3-engine params (required only in S3 mode).
    bucket: str = ""
    access_key: str = ""
    secret_key: str = ""
    scheme: str = "http"
    ca_bundle: str = ""

    def __post_init__(self) -> None:
        self.accelerated = _is_truthy(self.accelerated)
        if not self.endpoint_override:
            raise ValueError("ObjStoreConfig requires 'endpoint_override'")

        if self.accelerated:
            self._validate_accelerated()
        else:
            self._validate_s3()

    def _validate_accelerated(self) -> None:
        if not self.type:
            raise ValueError(
                "Accelerated object store tier requires 'type' "
                "(e.g. 'scality_ai_connector') and 'endpoint_override'."
            )
        mixed = [f for f in _S3_ONLY_FIELDS if getattr(self, f)]
        if mixed:
            raise ValueError(
                f"Accelerated object store tier must not set S3-only fields: "
                f"{', '.join(mixed)}. These are ignored by accelerated engines."
            )
        reserved = _RESERVED_ACCELERATED_PARAMS & set(self.extra_params)
        if reserved:
            raise ValueError(
                f"extra_params must not contain reserved keys: "
                f"{', '.join(sorted(reserved))}."
            )

    def _validate_s3(self) -> None:
        missing = [
            f for f in ("bucket", "access_key", "secret_key") if not getattr(self, f)
        ]
        if missing:
            raise ValueError(
                f"S3 object store tier requires: {', '.join(missing)}. "
                f"For an accelerated engine, set 'accelerated': true and 'type'."
            )

    def to_nixl_params(self) -> dict[str, str]:
        """Build the NIXL OBJ backend params dict."""
        if self.accelerated:
            params: dict[str, str] = {
                "accelerated": "true",
                "type": self.type,
                "endpoint_override": self.endpoint_override,
            }
            params.update({k: str(v) for k, v in self.extra_params.items()})
            return params

        params = {
            "bucket": self.bucket,
            "endpoint_override": self.endpoint_override,
            "scheme": self.scheme,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
        }
        if self.ca_bundle:
            params["ca_bundle"] = self.ca_bundle
        return params
