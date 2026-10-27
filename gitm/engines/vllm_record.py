"""Read a live vLLM engine without importing vLLM on CPU-only machines.

No independent HF reads, inferred architecture, tensor downloads, or tracing
hooks. Worker RPC is needed because V1 keeps the model in other processes.
This is an initial engine-description JSON, not a loop verdict schema.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.metadata
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_ROOTS = ("llm_engine.vllm_config", "engine.vllm_config", "vllm_config",
          "llm_engine", "engine", "")
_SECRET_KEYS = {"hf_token", "token", "api_key", "authorization", "password", "access_token"}
_HOOK_ENV = ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "ROCP_TOOL_LIBRARIES",
             "GITM_TRACE_OUT", "GITM_TRACE_NVTX")
_FIELDS = {
    "model": ("model", "revision", "tokenizer", "tokenizer_revision", "dtype",
              "quantization", "max_model_len", "enforce_eager", "is_mla", "is_moe"),
    "parallel": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "enable_expert_parallel", "world_size", "distributed_executor_backend"),
    "cache": ("cache_dtype", "block_size", "num_gpu_blocks", "gpu_memory_utilization",
              "enable_prefix_caching", "kv_cache_memory_bytes"),
    "scheduler": ("max_num_seqs", "max_num_batched_tokens", "chunked_prefill_enabled",
                  "async_scheduling", "policy"),
    "compilation": ("mode", "backend", "debug_dump_path", "cache_dir", "cudagraph_mode",
                    "cudagraph_capture_sizes", "compile_sizes"),
    "attention": ("backend",),
}


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_value(value: Any) -> Any:
    """Strict metadata-only conversion. Never repr arbitrary engine objects."""
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): "[redacted]" if str(k).lower() in _SECRET_KEYS else _json_value(v)
                for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(v) for v in value]
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    raise TypeError(f"unsupported metadata type: {_class_name(value)}")


def _at(obj: Any, path: str) -> Any:
    for part in path.split(".") if path else ():
        obj = obj[part] if isinstance(obj, Mapping) else getattr(obj, part)
    return obj


def _missing(source: str, reason: str) -> dict:
    return {"status": "unavailable", "value": None, "source": source, "reason": reason}


def _read(obj: Any, path: str, *, source: str | None = None) -> dict:
    source = source or path
    try:
        return {"status": "observed", "value": _json_value(_at(obj, path)), "source": source}
    except (AttributeError, KeyError):
        return _missing(source, "not exposed by this engine object")
    except Exception as exc:
        return _missing(source, f"read failed ({type(exc).__name__})")


def _find_config(engine: Any, section: str, *, include_none: bool = False) -> tuple[Any, str]:
    for prefix in _ROOTS:
        path = f"{prefix}.{section}" if prefix else section
        try:
            value = _at(engine, path)
            if value is not None or include_none:
                return value, path
        except (AttributeError, KeyError):
            continue
    return None, section


def _section(engine: Any, section: str, fields: tuple[str, ...]) -> dict:
    config, path = _find_config(engine, section)
    return {field: _read(config, field, source=f"{path}.{field}") for field in fields}


def _version() -> dict:
    try:
        return {"status": "observed", "value": importlib.metadata.version("vllm"),
                "source": "importlib.metadata.version('vllm')"}
    except importlib.metadata.PackageNotFoundError:
        return _missing("importlib.metadata.version('vllm')", "vLLM distribution not installed")


def _shape(engine: Any) -> dict:
    """The config vLLM already resolved, never an independent HF lookup."""
    cfg, path = _find_config(engine, "model_config")
    for field in ("hf_text_config", "hf_config"):
        try:
            value = _at(cfg, field)
            if value is None:
                continue
            # vLLM's resolved HF config is an engine object, not a file we load.
            data = (dict(value) if isinstance(value, Mapping) else value.to_dict()
                    if callable(getattr(value, "to_dict", None)) else vars(value))
            return {"status": "observed", "value": _json_value(data),
                    "source": f"{path}.{field}.to_dict()" if hasattr(value, "to_dict")
                    else f"{path}.{field}" if isinstance(value, Mapping)
                    else f"vars({path}.{field})"}
        except (AttributeError, KeyError, TypeError):
            continue
    return _missing(f"{path}.hf_text_config / hf_config", "resolved static config unavailable")


def _quantization(model: Any) -> dict:
    result = {"class": _missing("worker.get_model().quant_config", "not exposed")}
    config = getattr(model, "quant_config", None)
    if config is not None:
        result["class"] = {"status": "observed", "value": _class_name(config),
                           "source": "worker.get_model().quant_config.__class__"}
        for name in ("ignored_layers", "modules_to_not_convert", "weight_block_size",
                     "activation_scheme", "is_checkpoint_fp8_serialized"):
            if hasattr(config, name):
                result[name] = _read(config, name, source=f"worker.get_model().quant_config.{name}")
    return result


def _module_inventory(model: Any, limit: int) -> dict:
    """Loaded module classes, applied quant methods, and tensor *metadata* only."""
    rows = []
    errors = []
    for index, (name, module) in enumerate(model.named_modules()):
        if index >= limit:
            return {"status": "partial", "source": "worker.get_model().named_modules()",
                    "value": rows, "reason": f"module limit {limit} reached", "errors": errors}
        row = {"name": name, "class": _class_name(module)}
        backend = getattr(module, "get_attn_backend", None)
        if callable(backend):
            try:
                resolved = backend()
                row["attention_backend"] = {
                    "status": "observed", "source": f"module[{name!r}].get_attn_backend()",
                    "value": f"{resolved.__module__}.{resolved.__qualname__}"}
            except Exception as exc:
                errors.append({"module": name, "kind": "attention_backend", "error": type(exc).__name__})
        if hasattr(module, "quant_method"):
            method = module.quant_method
            row["quant_method"] = None if method is None else _class_name(method)
        tensors = []
        for kind, method in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
            try:
                for tensor_name, tensor in getattr(module, method)(recurse=False):
                    tensors.append({"name": tensor_name, "kind": kind,
                                    "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
            except Exception as exc:
                errors.append({"module": name, "kind": kind, "error": type(exc).__name__})
        row["tensors"] = tensors
        rows.append(row)
    return {"status": "partial" if errors else "observed", "value": rows,
            "source": "worker.get_model().named_modules(); named_parameters/buffers(recurse=False)",
            "errors": errors}


def inspect_worker(worker: Any, *, module_limit: int = 50000) -> dict:
    """vLLM collective_rpc callable: runs inside each worker after initialization.

    These memory fields are vLLM's profiling results, not sums of tensor numel.
    ``peak_activation_memory`` has engine-specific semantics (in 0.31 it also
    includes an applied CUDA-graph estimate); preserve the raw field name.
    """
    record = {
        "rank": _read(worker, "rank", source="worker.rank"),
        "local_rank": _read(worker, "local_rank", source="worker.local_rank"),
        "device": _read(worker, "device", source="worker.device"),
        "engine_version": _version(),
        "cache": _section(worker, "cache_config", _FIELDS["cache"]),
        "memory_bytes": {
            "model_memory_usage": _read(worker, "model_runner.model_memory_usage"),
            "peak_activation_memory": _read(worker, "peak_activation_memory"),
            "available_kv_cache_memory_bytes": _read(worker, "available_kv_cache_memory_bytes"),
            "cudagraph_memory_estimate": _read(worker, "cudagraph_memory_estimate"),
        },
        "kv_cache_config": {},
        "effective_kv_dtype": _read(worker, "model_runner.kv_cache_dtype"),
    }
    for name in ("num_blocks",):
        record["kv_cache_config"][name] = _read(worker, f"model_runner.kv_cache_config.{name}")
    try:
        model = worker.get_model()
        record["model_class"] = {"status": "observed", "value": _class_name(model),
                                 "source": "worker.get_model().__class__"}
        record["precision"] = _quantization(model)
        record["modules"] = _module_inventory(model, module_limit)
    except Exception as exc:
        record["modules"] = _missing("worker.get_model()", f"read failed ({type(exc).__name__})")
    return record


def inspect_graph_artifacts(directory: str | Path | None) -> dict:
    """Index readable graph text with hashes; no claim of graph/shape completeness."""
    if directory is None:
        return _missing("compilation debug dump", "no dump directory supplied")
    root = Path(directory).resolve()
    if not root.is_dir():
        return _missing(str(root), "dump directory does not exist")
    files, errors = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        # Index graph/debug text, not compiled binaries or tensor payloads.
        if path.suffix not in (".py", ".txt", ".json", ".log"):
            continue
        try:
            digest = hashlib.sha256()
            size = 0
            sample = b""
            decoder = codecs.getincrementaldecoder("utf-8")()
            with path.open("rb") as handle:
                while chunk := handle.read(65536):
                    decoder.decode(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if len(sample) < 65536:
                        sample += chunk[:65536 - len(sample)]
            decoder.decode(b"", final=True)
            text = sample.decode("utf-8", errors="replace")
            files.append({"path": str(path.relative_to(root)), "bytes": size,
                          "sha256": digest.hexdigest(),
                          "graph_candidate": any(marker in text for marker in
                                                 ("def forward", "GraphModule", "torch.ops"))})
        except (OSError, UnicodeError) as exc:
            errors.append({"path": str(path.relative_to(root)), "error": type(exc).__name__})
    candidates = sum(item["graph_candidate"] for item in files)
    status = "observed" if candidates and not errors else "partial" if files else "unavailable"
    return {"status": status, "source": str(root), "value": files, "errors": errors,
            "graph_candidates": candidates, "graph_completeness": "unverified",
            "shape_coverage": "unverified", "rank_coverage": "unverified"}


def _gaps(value: Any, path: str = "") -> list[dict]:
    out = []
    if isinstance(value, dict):
        if value.get("status") in ("unavailable", "partial"):
            out.append({"field": path, "reason": value.get("reason", "partial evidence")})
        for key, item in value.items():
            if key != "value":
                out.extend(_gaps(item, f"{path}.{key}".strip(".")))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(_gaps(item, f"{path}[{index}]"))
    return out


def extract_vllm_record(engine: Any, *, graph_dir: str | Path | None = None,
                        rpc_timeout_s: float = 60.0) -> dict:
    """Describe a live LLM/LLMEngine; existing HTTP endpoints alone aren't assumed sufficient."""
    result = {
        "schema_version": "gitm.engine-description.v1",
        "engine": "vllm", "engine_version": _version(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "model": _section(engine, "model_config", _FIELDS["model"]),
        "static_config": _shape(engine),
        "deployment": {name: _section(engine, f"{name}_config", _FIELDS[name])
                       for name in ("parallel", "cache", "scheduler", "attention")},
        "compilation": _section(engine, "compilation_config", _FIELDS["compilation"]),
        "graph_artifacts": inspect_graph_artifacts(graph_dir),
    }
    model_config, model_source = _find_config(engine, "model_config")
    result["model"]["resolved_config_commit"] = _read(
        model_config, "hf_config._commit_hash", source=f"{model_source}.hf_config._commit_hash")
    speculative, source = _find_config(engine, "speculative_config", include_none=True)
    if speculative is None:
        # Distinguish an explicit None (disabled) from a missing attribute.
        result["speculative"] = _read(engine, source)
    else:
        result["speculative"] = {name: _read(speculative, name, source=f"{source}.{name}")
                                 for name in ("method", "num_speculative_tokens")}
    rpc = getattr(engine, "collective_rpc", None)
    if callable(rpc):
        try:
            workers = rpc(inspect_worker, timeout=rpc_timeout_s)
            if not isinstance(workers, list) or not workers or any(
                    not isinstance(w, dict) for w in workers):
                raise ValueError("worker RPC returned no usable records")
            result["workers"] = workers
        except Exception as exc:
            result["workers"] = _missing("vllm.collective_rpc(inspect_worker)",
                                          f"RPC failed ({type(exc).__name__})")
    else:
        result["workers"] = _missing("vllm.collective_rpc(inspect_worker)", "RPC not exposed")
    result["gaps"] = _gaps(result)
    requested = result["model"]["revision"]["value"]
    resolved = result["model"]["resolved_config_commit"]["value"]
    if requested and resolved and requested != resolved:
        result["gaps"].append({"field": "model.resolved_config_commit",
                               "reason": "requested and resolved configuration commits differ"})
    workers = result["workers"]
    if isinstance(workers, list):
        expected = result["deployment"]["parallel"]["world_size"]["value"]
        ranks = [w.get("rank", {}).get("value") for w in workers]
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            result["gaps"].append({"field": "workers", "reason": "expected worker count unavailable"})
        elif len(workers) != expected or sorted(
                r for r in ranks if isinstance(r, int) and not isinstance(r, bool)) != list(range(expected)):
            result["gaps"].append({"field": "workers", "reason": "worker count/ranks do not cover engine world_size"})
        for index, worker in enumerate(workers):
            if worker.get("engine_version", {}).get("value") != result["engine_version"]["value"]:
                result["gaps"].append({"field": f"workers[{index}].engine_version",
                                      "reason": "worker and driver versions differ"})
            for name in ("model_memory_usage", "peak_activation_memory", "available_kv_cache_memory_bytes"):
                value = worker.get("memory_bytes", {}).get(name, {}).get("value")
                if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                    result["gaps"].append({"field": f"workers[{index}].memory_bytes.{name}",
                                          "reason": "post-initialization memory measurement unavailable"})
    result["status"] = "partial" if result["gaps"] else "extracted"
    # Extraction never establishes deployment acceptance, graph completeness,
    # model correctness, or planner parity by itself.
    result["acceptance"] = "pending_live_review"
    return result


def extend_run_record(run_record: Mapping[str, Any], engine_record: Mapping[str, Any],
                      *, instance_id: str) -> dict:
    """Add engine evidence to ENG-72's record without changing its verdict fields.

    ``gitm.engine_descriptions`` is a proposed extension, keyed by engine instance
    so a control and a canary need not share one configuration. The caller owns
    the association with canary_instance/control_instances (or their namespaced
    equivalents). An empty base produces an extension fragment, not a completed
    loop record. Never manufacture gain, noise-floor, health, or policy values.
    """
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a nonempty string")
    key = "gitm.engine_descriptions"
    result = deepcopy(dict(run_record))
    descriptions = result.get(key, {})
    if not isinstance(descriptions, dict):
        raise ValueError(f"{key} must be a mapping")
    payload = _json_value(engine_record)
    if instance_id in descriptions and descriptions[instance_id] != payload:
        raise ValueError("engine instance already has different evidence; preserve the existing observation")
    descriptions[instance_id] = deepcopy(payload)
    result[key] = descriptions
    return result


def _check_launch_environment() -> None:
    active = [key for key in _HOOK_ENV if os.environ.get(key)]
    if active:
        raise ValueError("run without tracing injection; unset: " + ", ".join(active))


def main(argv: list[str] | None = None) -> int:
    """Opt-in GPU launch probe; vLLM parses its own engine arguments after --."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-vllm-version", required=True)
    parser.add_argument("--rpc-timeout", type=float, default=60.0)
    parser.add_argument("--instance-id", help="engine instance label; default: a fresh probe-only UUID")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    launch = args.engine_args[1:] if args.engine_args[:1] == ["--"] else args.engine_args
    if not launch:
        parser.error("supply vLLM engine arguments after --, including --model")
    try:
        _check_launch_environment()
        installed = importlib.metadata.version("vllm")
        if installed != args.expected_vllm_version:
            raise ValueError(f"expected vLLM {args.expected_vllm_version}; installed {installed}")
        if args.rpc_timeout <= 0 or not math.isfinite(args.rpc_timeout):
            raise ValueError("RPC timeout must be positive and finite")
        if args.instance_id is not None and not args.instance_id.strip():
            raise ValueError("instance_id must be a nonempty string")
        if args.out.exists():
            raise ValueError("output directory already exists; use a fresh path for this observation")
        from vllm import LLM, SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        vparser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        config = EngineArgs.from_cli_args(vparser.parse_args(launch))
        if not isinstance(config.revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", config.revision):
            raise ValueError("supply --revision with the team's immutable 40-character checkpoint SHA")
        mode = config.compilation_config.mode
        if config.enforce_eager or (mode is not None and int(mode) == 0):
            raise ValueError("first check requires compilation enabled; preserve the serving config")
        if getattr(config, "enable_layerwise_nvtx_tracing", False):
            raise ValueError("first check must not enable layerwise tracing hooks")
        dump = args.out / "compilation"
        if config.compilation_config.debug_dump_path is not None:
            raise ValueError("omit debug_dump_path; the probe uses a fresh directory under --out")
        config.compilation_config.debug_dump_path = dump
        args.out.mkdir(parents=True)
        engine = LLM.from_engine_args(config)
        try:
            # Exercise the compiled forward path, with deterministic token IDs;
            # this is a graph check, not a performance or output-quality test.
            engine.generate([{"prompt_token_ids": [1, 2, 3, 4]}],
                            SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
            record = extract_vllm_record(engine, graph_dir=dump, rpc_timeout_s=args.rpc_timeout)
            record["probe"] = {"kind": "in_process_launch", "gitm_tracing_hooks": False,
                               "image_identity": "unverified", "http_serving_parity": "unverified",
                               "note": "One generated token exercises compilation; not a benchmark."}
            instance_id = args.instance_id or f"probe-{uuid.uuid4().hex}"
            extension = extend_run_record({}, record, instance_id=instance_id)
            (args.out / "engine_record.json").write_text(json.dumps(extension, indent=2, allow_nan=False))
        finally:
            core = getattr(engine.llm_engine, "engine_core", None)
            shutdown = getattr(core, "shutdown", None)
            if callable(shutdown):
                shutdown()
        print(f"Run-record engine extension: {args.out / 'engine_record.json'}")
        print(f"Extraction: {record['status']}; {len(record['gaps'])} gaps; live review pending")
        return 0 if record["graph_artifacts"].get("graph_candidates", 0) else 2
    except (ValueError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(2, f"engine extraction: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
