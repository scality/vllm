# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# NIXL's S3 plugin. Any other backend exposing OBJ_SEG (e.g. `REST`,
# `AZURE_BLOB`) is configured only through `endpoint_override` and
# `backend_params`.
S3_BACKEND = "OBJ"

_S3_ONLY_FIELDS = (
    "bucket",
    "access_key",
    "secret_key",
    "session_token",
    "region",
    "ca_bundle",
)

_RESERVED_BACKEND_PARAM_KEYS = frozenset(
    {
        "bucket",
        "endpoint_override",
        "access_key",
        "secret_key",
        "session_token",
        "region",
        "scheme",
        "ca_bundle",
        "num_threads",
    }
)


def _to_nixl_param(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower()
    return str(value)


@dataclass
class ObjStoreConfig:
    """Connection parameters for an object store backend.

    `backend` names the NIXL plugin. The default, `OBJ`, is the S3 plugin
    and requires `bucket` and `endpoint_override`. When `access_key` and
    `secret_key` are left empty it falls back to the AWS SDK default
    credential provider chain (IAM roles, environment variables, credential
    files, etc.), which enables workload-identity based auth on Kubernetes.

    Any other `backend` must expose object segments (OBJ_SEG); it receives
    `endpoint_override` (when set) and `backend_params` only.
    """

    backend: str = S3_BACKEND
    bucket: str = ""
    endpoint_override: str = ""
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    session_token: str = field(default="", repr=False)
    region: str = ""
    scheme: str = "http"
    ca_bundle: str = ""
    backend_params: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("ObjStoreConfig requires a non-empty 'backend'")
        if self.backend == S3_BACKEND:
            missing = [
                f for f in ("bucket", "endpoint_override") if not getattr(self, f)
            ]
            if missing:
                raise ValueError(
                    f"Object store tier with backend {S3_BACKEND!r} requires: "
                    f"{', '.join(missing)}"
                )
            return
        s3_fields = [f for f in _S3_ONLY_FIELDS if getattr(self, f)]
        if s3_fields:
            raise ValueError(
                f"Object store tier with backend {self.backend!r} must not set "
                f"fields of the {S3_BACKEND!r} backend: {', '.join(s3_fields)}"
            )

    def to_nixl_params(self) -> dict[str, str]:
        """Build the params dict for the NIXL backend named by `backend`.

        For the S3 backend, credential and optional fields are only included
        when non-empty so that the AWS SDK default credential chain can
        activate.
        """
        duplicate_keys = _RESERVED_BACKEND_PARAM_KEYS.intersection(self.backend_params)
        if duplicate_keys:
            raise ValueError(
                "backend_params cannot override reserved object store "
                f"parameters: {sorted(duplicate_keys)}"
            )

        params: dict[str, str] = {}
        if self.backend != S3_BACKEND:
            if self.endpoint_override:
                params["endpoint_override"] = self.endpoint_override
        else:
            params = {
                "bucket": self.bucket,
                "endpoint_override": self.endpoint_override,
                "scheme": self.scheme,
            }
            # Omit empty optional fields so the NIXL OBJ plugin's underlying
            # AWS SDK can fall back to its default credential provider chain
            # (IAM roles, env vars, credential files, etc.).
            # https://github.com/ai-dynamo/nixl/blob/main/src/plugins/obj/README.md
            for key in (
                "access_key",
                "secret_key",
                "session_token",
                "region",
                "ca_bundle",
            ):
                value = getattr(self, key)
                if value:
                    params[key] = value

        params.update(
            {key: _to_nixl_param(value) for key, value in self.backend_params.items()}
        )
        return params
