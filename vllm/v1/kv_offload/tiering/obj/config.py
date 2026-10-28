# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from dataclasses import dataclass, field

# NIXL's S3 plugin. Any other backend that exposes OBJ_SEG (e.g. ``REST``,
# ``AZURE_BLOB``) is configured only through ``endpoint_override`` and
# ``extra_params``.
_S3_BACKEND = "OBJ"

# NIXL backend params that the accelerated engine owns or that the manager
# injects (``num_threads`` from ``io_threads``). They may not be overridden
# through ``extra_params``.
_RESERVED_ACCELERATED_PARAMS = frozenset(
    {"accelerated", "type", "endpoint_override", "num_threads"}
)

# S3-only params that are meaningless to an accelerated engine. Supplying any
# of them alongside ``accelerated`` indicates a mixed-mode config.
_S3_ONLY_FIELDS = ("bucket", "access_key", "secret_key", "session_token", "region")


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

    Three modes are supported, the first two on NIXL's S3 plugin (``OBJ``):

    - **S3 mode** (default): drives NIXL's default AWS-SDK S3 engine over
      HTTP(S). Requires ``bucket`` and ``endpoint_override``. When
      ``access_key`` and ``secret_key`` are left empty the NIXL OBJ plugin
      falls back to the AWS SDK default credential provider chain (IAM roles,
      environment variables, credential files, etc.), which enables
      workload-identity based auth on Kubernetes.
    - **Accelerated mode** (``accelerated=true``): selects a NIXL accelerated
      OBJ engine (e.g. ``scality_ai_connector``, ``dell``) that moves bytes
      over cuObject/GPUDirect RDMA. Requires ``type`` and ``endpoint_override``;
      ``bucket``/``access_key``/``secret_key``/``scheme`` are not used.
      Engine-specific params may be passed through ``extra_params``.
    - **Other OBJ_SEG backend** (``backend`` other than ``OBJ``, e.g.
      ``REST`` for the Scality AI Connector): any NIXL plugin that exposes
      object segments. ``endpoint_override`` (when set) and ``extra_params``
      are passed through as its params; S3 fields, ``accelerated`` and
      ``type`` are not used.
    """

    backend: str = _S3_BACKEND
    endpoint_override: str = ""
    # Accelerated-engine params.
    accelerated: bool = False
    type: str = ""
    extra_params: dict[str, str] = field(default_factory=dict)
    # S3-engine params (required only in S3 mode).
    bucket: str = ""
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    session_token: str = field(default="", repr=False)
    region: str = ""
    scheme: str = "http"
    ca_bundle: str = ""

    def __post_init__(self) -> None:
        self.accelerated = _is_truthy(self.accelerated)
        if not self.backend:
            raise ValueError("ObjStoreConfig requires a non-empty 'backend'")

        if self.backend != _S3_BACKEND:
            self._validate_other_backend()
            return
        if not self.endpoint_override:
            raise ValueError("ObjStoreConfig requires 'endpoint_override'")
        if self.accelerated:
            self._validate_accelerated()
        else:
            self._validate_s3()

    def _validate_other_backend(self) -> None:
        if self.accelerated or self.type:
            raise ValueError(
                f"Object store tier with backend {self.backend!r} must not set "
                f"'accelerated' or 'type'; those select an engine of the "
                f"{_S3_BACKEND} backend."
            )
        self._validate_passthrough_params(self.backend)

    def _validate_accelerated(self) -> None:
        if not self.type:
            raise ValueError(
                "Accelerated object store tier requires 'type' "
                "(e.g. 'scality_ai_connector') and 'endpoint_override'."
            )
        self._validate_passthrough_params("Accelerated")

    def _validate_passthrough_params(self, mode: str) -> None:
        mixed = [f for f in _S3_ONLY_FIELDS if getattr(self, f)]
        if mixed:
            raise ValueError(
                f"{mode} object store tier must not set S3-only fields: "
                f"{', '.join(mixed)}. These are only used by the S3 engine."
            )
        reserved = _RESERVED_ACCELERATED_PARAMS & set(self.extra_params)
        if reserved:
            raise ValueError(
                f"extra_params must not contain reserved keys: "
                f"{', '.join(sorted(reserved))}."
            )

    def _validate_s3(self) -> None:
        if not self.bucket:
            raise ValueError(
                "S3 object store tier requires 'bucket'. "
                "For an accelerated engine, set 'accelerated': true and 'type'."
            )

    def to_nixl_params(self) -> dict[str, str]:
        """Build the params dict for the NIXL backend named by ``backend``."""
        if self.backend != _S3_BACKEND:
            params: dict[str, str] = {}
            if self.endpoint_override:
                params["endpoint_override"] = self.endpoint_override
            params.update({k: str(v) for k, v in self.extra_params.items()})
            return params

        if self.accelerated:
            params = {
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
        }
        # Omit empty optional fields so the NIXL OBJ plugin's underlying
        # AWS SDK can fall back to its default credential provider chain
        # (IAM roles, env vars, credential files, etc.).
        # https://github.com/ai-dynamo/nixl/blob/main/src/plugins/obj/README.md
        for key in ("access_key", "secret_key", "session_token", "region", "ca_bundle"):
            value = getattr(self, key)
            if value:
                params[key] = value
        return params
