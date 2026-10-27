"""CPU contracts for extraction. No real engine or GPU is constructed here."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from gitm.engines import vllm_record as record


@pytest.fixture(autouse=True)
def isolated_engine_environment(monkeypatch):
    # Other trace tests may leave injection enabled. Each probe test controls
    # its own launch environment; the rejection test explicitly sets its hook.
    for key in record._HOOK_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.31.0")


def _config(world=2):
    return NS(
        model_config=NS(model="moonshotai/Kimi-K2.6", revision="a" * 40, dtype="bfloat16",
                        hf_text_config=NS(hidden_size=7168, n_routed_experts=384)),
        parallel_config=NS(world_size=world, tensor_parallel_size=world),
        speculative_config=None,
        compilation_config=NS(mode=3),
    )


class _Model:
    quant_config = NS(ignored_layers=["lm_head"], weight_block_size=[128, 128])

    def named_modules(self):
        yield "lm_head", self

    def named_parameters(self, recurse=False):
        assert not recurse
        yield "weight", NS(shape=(8, 16), dtype="torch.bfloat16")

    def named_buffers(self, recurse=False):
        return iter(())


def _worker(rank=0):
    return NS(rank=rank, local_rank=rank, device=f"cuda:{rank}",
              model_runner=NS(model_memory_usage=123456 + rank,
                              kv_cache_dtype="torch.bfloat16", kv_cache_config=NS(num_blocks=42)),
              peak_activation_memory=1024, available_kv_cache_memory_bytes=8192,
              cache_config=NS(num_gpu_blocks=42, block_size=16), get_model=_Model)


def _engine(world=2):
    def rpc(fn, timeout):
        assert timeout == 60.0
        return [fn(_worker(rank)) for rank in range(world)]
    return NS(llm_engine=NS(vllm_config=_config(world)), collective_rpc=rpc)


@pytest.mark.parametrize("root", ["llm_engine.vllm_config", "engine.vllm_config", "vllm_config", ""])
def test_engine_resolved_values_have_exact_sources_without_hf_reads(root):
    engine = _config()
    for part in reversed(root.split(".")) if root else ():
        engine = NS(**{part: engine})
    result = record.extract_vllm_record(engine)
    assert result["model"]["revision"]["value"] == "a" * 40
    prefix = root + "." if root else ""
    assert result["model"]["revision"]["source"] == prefix + "model_config.revision"
    assert result["static_config"]["value"]["n_routed_experts"] == 384
    assert result["speculative"]["status"] == "observed"
    assert result["speculative"]["value"] is None  # disabled, not missing
    json.dumps(result, allow_nan=False)


def test_missing_config_never_becomes_a_default_model_or_precision():
    result = record.extract_vllm_record(object())
    assert result["model"]["dtype"]["status"] == "unavailable"
    assert result["model"]["dtype"]["value"] is None
    assert result["static_config"]["status"] == "unavailable"
    assert result["status"] == "partial"
    assert result["acceptance"] == "pending_live_review"


def test_memory_is_engine_profiled_and_rank_local_not_a_weight_formula():
    result = record.extract_vllm_record(_engine())
    assert [w["rank"]["value"] for w in result["workers"]] == [0, 1]
    assert [w["memory_bytes"]["model_memory_usage"]["value"] for w in result["workers"]] == [123456, 123457]
    # Tensor shape would give 256 bytes in bf16, deliberately unlike 123456.
    assert result["workers"][0]["modules"]["value"][0]["tensors"][0]["shape"] == [8, 16]
    assert result["workers"][0]["cache"]["num_gpu_blocks"]["value"] == 42
    assert result["workers"][0]["effective_kv_dtype"]["value"] == "torch.bfloat16"
    assert result["workers"][0]["precision"]["ignored_layers"]["value"] == ["lm_head"]
    assert not any(g["reason"] == "worker count/ranks do not cover engine world_size" for g in result["gaps"])


def test_module_precision_distinguishes_applied_quant_method_and_tensor_dtype():
    model = _Model()
    model.quant_method = NS()
    worker = _worker()
    worker.get_model = lambda: model
    row = record.inspect_worker(worker)["modules"]["value"][0]
    assert row["quant_method"] == "types.SimpleNamespace"
    assert row["tensors"][0]["dtype"] == "torch.bfloat16"


def test_worker_rpc_failure_remains_a_gap():
    def fail(*args, **kwargs):
        raise TimeoutError("sensitive exception text must not be copied")
    result = record.extract_vllm_record(NS(collective_rpc=fail))
    assert result["workers"]["status"] == "unavailable"
    assert "TimeoutError" in result["workers"]["reason"]
    assert "sensitive exception" not in json.dumps(result)


@pytest.mark.parametrize("ranks", [[0], [0, 0], [0, 2]])
def test_missing_duplicate_or_wrong_ranks_block_complete_extraction(ranks):
    engine = _engine()
    engine.collective_rpc = lambda fn, timeout: [fn(_worker(r)) for r in ranks]
    result = record.extract_vllm_record(engine)
    assert any(g["reason"] == "worker count/ranks do not cover engine world_size" for g in result["gaps"])


def test_worker_version_mismatch_is_visible():
    engine = _engine(1)
    def rpc(fn, timeout):
        worker = fn(_worker())
        worker["engine_version"]["value"] = "0.30.0"
        return [worker]
    engine.collective_rpc = rpc
    result = record.extract_vllm_record(engine)
    assert any(g["reason"] == "worker and driver versions differ" for g in result["gaps"])


def test_missing_profile_is_not_replaced_by_nvml_or_zero():
    worker = _worker()
    del worker.model_runner.model_memory_usage
    result = record.inspect_worker(worker)
    assert result["memory_bytes"]["model_memory_usage"]["status"] == "unavailable"
    assert result["memory_bytes"]["model_memory_usage"]["value"] is None


def test_resolved_checkpoint_commit_is_separate_from_requested_revision():
    engine = _engine()
    engine.llm_engine.vllm_config.model_config.hf_config = NS(_commit_hash="b" * 40)
    result = record.extract_vllm_record(engine)
    assert result["model"]["revision"]["value"] == "a" * 40
    assert result["model"]["resolved_config_commit"]["value"] == "b" * 40
    assert any(g["reason"] == "requested and resolved configuration commits differ" for g in result["gaps"])


def test_resolved_attention_backend_is_read_from_the_loaded_module():
    class Backend:
        pass
    model = _Model()
    model.get_attn_backend = lambda: Backend
    worker = _worker()
    worker.get_model = lambda: model
    item = record.inspect_worker(worker)["modules"]["value"][0]["attention_backend"]
    assert item["status"] == "observed"
    assert item["value"].endswith("Backend")


def test_inventory_limit_is_explicit():
    result = record.inspect_worker(_worker(), module_limit=0)
    assert result["modules"]["status"] == "partial"
    assert "limit" in result["modules"]["reason"]


def test_graph_index_checks_content_and_hashes_without_claiming_complete_graph(tmp_path):
    graph = tmp_path / "rank_0" / "computation_graph.py"
    graph.parent.mkdir()
    payload = b"class GraphModule:\n    def forward(self, x): return x\n"
    graph.write_bytes(payload)
    (tmp_path / "unrelated.log").write_text("server ready")
    (tmp_path / "weights.bin").write_bytes(b"not graph text")
    (tmp_path / "link.py").symlink_to(graph)
    result = record.inspect_graph_artifacts(tmp_path)
    assert result["status"] == "observed"
    assert result["graph_candidates"] == 1
    item = next(f for f in result["value"] if f["path"].endswith("computation_graph.py"))
    assert item["sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(result["value"]) == 2
    assert result["graph_completeness"] == result["shape_coverage"] == "unverified"


def test_artifact_checks_utf8_beyond_first_chunk(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"def forward(): pass\n" + b"x" * 70000 + b"\xff")
    result = record.inspect_graph_artifacts(tmp_path)
    assert result["status"] == "unavailable"
    assert result["errors"][0]["error"] == "UnicodeDecodeError"


def test_utf8_character_split_across_chunk_is_valid(tmp_path):
    (tmp_path / "valid.py").write_bytes(b"def forward(): pass\n".ljust(65535, b" ") + "é".encode())
    assert record.inspect_graph_artifacts(tmp_path)["status"] == "observed"


def test_logs_alone_do_not_prove_a_graph_was_dumped(tmp_path):
    (tmp_path / "server.log").write_text("healthy")
    result = record.inspect_graph_artifacts(tmp_path)
    assert result["graph_candidates"] == 0
    assert result["status"] == "partial"


def test_missing_or_nonexistent_dump_is_a_gap(tmp_path):
    assert record.inspect_graph_artifacts(None)["status"] == "unavailable"
    assert record.inspect_graph_artifacts(tmp_path / "absent")["status"] == "unavailable"


def test_nested_credentials_are_redacted():
    engine = _engine()
    config = engine.llm_engine.vllm_config.model_config.hf_text_config
    config.extra = {"hf_token": "never-save-this", "input_tokens": 128}
    result = record.extract_vllm_record(engine)
    assert "never-save-this" not in json.dumps(result)
    assert result["static_config"]["value"]["extra"]["input_tokens"] == 128


def test_nonfinite_or_arbitrary_objects_are_not_silently_stringified():
    assert record._read(NS(x=float("nan")), "x")["status"] == "unavailable"
    assert record._read(NS(x=object()), "x")["status"] == "unavailable"


def test_probe_refuses_tracing_before_importing_vllm(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/already-injected.so")
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(SystemExit) as exc:
        record.main(["--out", str(tmp_path / "probe"), "--expected-vllm-version", "0.31.0",
                     "--", "--model", "example"])
    assert exc.value.code == 2
    assert "unset: CUDA_INJECTION64_PATH" in capsys.readouterr().err
    assert not (tmp_path / "probe").exists()


def test_probe_refuses_version_drift_before_engine_construction(tmp_path, capsys):
    with pytest.raises(SystemExit):
        record.main(["--out", str(tmp_path / "probe"), "--expected-vllm-version", "0.30.0",
                     "--", "--model", "example"])
    assert "installed 0.31.0" in capsys.readouterr().err


def _fake_vllm(monkeypatch, tmp_path, *, mode=3, revision="a" * 40):
    events = []
    class FakeEngineArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model")
            return parser

        @staticmethod
        def from_cli_args(ns):
            return NS(model=ns.model, revision=revision, enforce_eager=False,
                      compilation_config=NS(mode=mode, debug_dump_path=None))

    class FakeLLM:
        @staticmethod
        def from_engine_args(args):
            events.append("build")
            dump = args.compilation_config.debug_dump_path
            dump.mkdir()
            (dump / "graph.py").write_text("def forward(x): return x")
            engine = _engine(1)
            engine.generate = lambda *a, **kw: events.append("generate")
            engine.llm_engine.engine_core = NS(shutdown=lambda: events.append("shutdown"))
            return engine

    for name in ("vllm", "vllm.engine", "vllm.engine.arg_utils", "vllm.utils", "vllm.utils.argparse_utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm"].LLM = FakeLLM
    sys.modules["vllm"].SamplingParams = lambda **kw: kw
    sys.modules["vllm.engine.arg_utils"].EngineArgs = FakeEngineArgs
    import argparse
    sys.modules["vllm.utils.argparse_utils"].FlexibleArgumentParser = argparse.ArgumentParser
    return events


def test_probe_exercises_compilation_writes_json_and_shuts_down(tmp_path, monkeypatch):
    events = _fake_vllm(monkeypatch, tmp_path)
    out = tmp_path / "probe"
    assert record.main(["--out", str(out), "--expected-vllm-version", "0.31.0",
                        "--", "--model", "moonshotai/Kimi-K2.6"]) == 0
    assert events == ["build", "generate", "shutdown"]
    data = json.loads((out / "engine_record.json").read_text())
    assert list(data) == ["gitm.engine_descriptions"]
    key, payload = next(iter(data["gitm.engine_descriptions"].items()))
    assert key.startswith("probe-")
    assert payload["probe"]["gitm_tracing_hooks"] is False
    assert payload["acceptance"] == "pending_live_review"


def test_probe_releases_engine_even_when_recording_fails(tmp_path, monkeypatch):
    events = _fake_vllm(monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("recording failed")
    monkeypatch.setattr(record, "extract_vllm_record", fail)
    with pytest.raises(RuntimeError, match="recording failed"):
        record.main(["--out", str(tmp_path / "probe"), "--expected-vllm-version", "0.31.0",
                     "--", "--model", "example"])
    assert events == ["build", "generate", "shutdown"]


@pytest.mark.parametrize("mode,revision", [(0, "a" * 40), (3, "main")])
def test_probe_refuses_eager_or_unpinned_revision_before_build(tmp_path, monkeypatch, mode, revision):
    events = _fake_vllm(monkeypatch, tmp_path, mode=mode, revision=revision)
    with pytest.raises(SystemExit):
        record.main(["--out", str(tmp_path / "probe"), "--expected-vllm-version", "0.31.0",
                     "--", "--model", "example"])
    assert events == []


def test_extension_preserves_existing_eng72_verdict_and_provenance_fields():
    base = {
        "beat_baseline": False, "delta": -0.01, "iterations": 2, "wall_time_s": 120,
        "node_healthy_for_window": False, "baseline_config": {"tensor_parallel_size": 8},
        "candidate_config": {"tensor_parallel_size": 8, "max_num_seqs": 64},
        "canary_instance": "candidate", "control_instances": ["baseline"],
        "policy_id": "p1", "degradations": ["existing-gap"], "state": "invalid",
    }
    out = record.extend_run_record(base, {"model": "kimi"}, instance_id="candidate")
    assert {k: out[k] for k in base} == base
    assert "gitm.engine_descriptions" not in base
    out["baseline_config"]["tensor_parallel_size"] = 2
    assert base["baseline_config"]["tensor_parallel_size"] == 8


def test_extension_keeps_control_and_canary_evidence_separate():
    first = record.extend_run_record({}, {"precision": "bf16"}, instance_id="control")
    second = record.extend_run_record(first, {"precision": "fp8"}, instance_id="canary")
    assert set(second["gitm.engine_descriptions"]) == {"control", "canary"}
    assert set(first["gitm.engine_descriptions"]) == {"control"}
    assert second["gitm.engine_descriptions"]["control"]["precision"] == "bf16"


def test_empty_base_is_only_an_extension_not_a_fabricated_loop_verdict():
    out = record.extend_run_record({}, {"acceptance": "pending_live_review"}, instance_id="probe")
    assert set(out) == {"gitm.engine_descriptions"}
    for name in ("beat_baseline", "delta", "node_healthy_for_window", "state", "run_id"):
        assert name not in out


def test_extension_refuses_to_overwrite_existing_instance_evidence():
    out = record.extend_run_record({}, {"engine_version": "0.31.0"}, instance_id="same")
    assert record.extend_run_record(out, {"engine_version": "0.31.0"}, instance_id="same") == out
    with pytest.raises(ValueError, match="different evidence"):
        record.extend_run_record(out, {"engine_version": "0.32.0"}, instance_id="same")


@pytest.mark.parametrize("instance", ["", "   ", None])
def test_extension_needs_an_instance_label(instance):
    with pytest.raises(ValueError, match="nonempty"):
        record.extend_run_record({}, {}, instance_id=instance)


def test_extension_preserves_namespaced_pamela_record_keys_too():
    base = {"gitm.beat_baseline": True, "gitm.node_healthy_for_window": False}
    out = record.extend_run_record(base, {}, instance_id="engine")
    assert out["gitm.beat_baseline"] is True
    assert out["gitm.node_healthy_for_window"] is False
