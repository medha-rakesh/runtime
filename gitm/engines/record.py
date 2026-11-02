"""Engine-neutral record helpers shared across engine extractors.

Every engine extractor builds the same things: ``{status, value, source}``
evidence fields, a strict JSON-safe metadata conversion that redacts secrets and
refuses to stringify arbitrary objects, a gap collector, and the ENG-72 run
record extension. That engine-neutral logic lives here so a second engine
adapter (SGLang, later) does not restate it. Engine-specific config navigation
— where vLLM keeps ``vllm_config``, which worker fields exist — stays in each
engine's own module.

This is deliberately a handful of small functions, not a base class or a shared
engine protocol. The shared engine *interface* is ENG-69/ENG-73's to define;
this module only factors out the serialization and validation the vLLM extractor
already needed, so the next engine reuses it instead of copying it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

#: Mapping keys whose values are never serialized into an engine record.
SECRET_KEYS = {"hf_token", "token", "api_key", "authorization", "password", "access_token"}


def class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def json_value(value: Any) -> Any:
    """Strict metadata-only conversion. Never repr arbitrary engine objects."""
    if isinstance(value, Enum):
        return json_value(value.value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): "[redacted]" if str(k).lower() in SECRET_KEYS else json_value(v)
                for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(v) for v in value]
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    raise TypeError(f"unsupported metadata type: {class_name(value)}")


def navigate(obj: Any, path: str) -> Any:
    """Walk a dotted attribute/key path into an engine object or mapping."""
    for part in path.split(".") if path else ():
        obj = obj[part] if isinstance(obj, Mapping) else getattr(obj, part)
    return obj


def missing(source: str, reason: str) -> dict:
    return {"status": "unavailable", "value": None, "source": source, "reason": reason}


def observe(obj: Any, path: str, *, source: str | None = None) -> dict:
    """Read one metadata field as a ``{status, value, source}`` evidence dict."""
    source = source or path
    try:
        return {"status": "observed", "value": json_value(navigate(obj, path)), "source": source}
    except (AttributeError, KeyError):
        return missing(source, "not exposed by this engine object")
    except Exception as exc:
        return missing(source, f"read failed ({type(exc).__name__})")


def collect_gaps(value: Any, path: str = "") -> list[dict]:
    """Walk a record and list every field left unavailable or partial."""
    out = []
    if isinstance(value, dict):
        if value.get("status") in ("unavailable", "partial"):
            out.append({"field": path, "reason": value.get("reason", "partial evidence")})
        for key, item in value.items():
            if key != "value":
                out.extend(collect_gaps(item, f"{path}.{key}".strip(".")))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(collect_gaps(item, f"{path}[{index}]"))
    return out


def extend_run_record(run_record: Mapping[str, Any], engine_record: Mapping[str, Any],
                      *, instance_id: str) -> dict:
    """Add engine evidence to ENG-72's record without changing its verdict fields.

    ``gitm.engine_descriptions`` is a proposed extension, keyed by engine instance
    so a control and a canary need not share one configuration. The caller owns
    the association with canary_instance/control_instances (or their namespaced
    equivalents). An empty base produces an extension fragment, not a completed
    loop record. Never manufacture gain, noise-floor, health, or policy values.

    This helper is engine-neutral: it takes an already-built engine record from
    any extractor and attaches it, so the vLLM and (later) SGLang paths share one
    serialization and overwrite-protection rule rather than two.
    """
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a nonempty string")
    key = "gitm.engine_descriptions"
    result = deepcopy(dict(run_record))
    descriptions = result.get(key, {})
    if not isinstance(descriptions, dict):
        raise ValueError(f"{key} must be a mapping")
    payload = json_value(engine_record)
    if instance_id in descriptions and descriptions[instance_id] != payload:
        raise ValueError("engine instance already has different evidence; preserve the existing observation")
    descriptions[instance_id] = deepcopy(payload)
    result[key] = descriptions
    return result
