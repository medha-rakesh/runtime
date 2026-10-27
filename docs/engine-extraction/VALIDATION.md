# ENG-85 validation — October 27, 2026

Base: runtime main `bbc6c07587cece4dfa42ef6a282bcb253fc2a942`.

## Local verification

```bash
python -m pytest tests/test_vllm_engine_record.py tests/test_vllm_stats_v1.py \
  tests/test_predict_graph_from_engine.py tests/test_gpu_detect.py
# Match .github/workflows/tests.yml (a newer Ruff no longer checks UP038).
ruff --version  # 0.12.11
ruff check .
bash -n deploy/h200/eng85-extract.sh
python -m gitm.engines.vllm_record --help
git diff --check
```

Result: **71 CPU tests passed** (37 new extractor/record-extension cases and 34 existing
config/planner regression cases). Ruff, shell syntax, CPU-only CLI help and
whitespace checks passed. A fresh JUnit report is included in the delivery package.
The probe tests construct fake vLLM objects; they do not run vLLM or use a GPU.

Behavior covered: config layouts and sources, missing/disabled distinction,
rank-local measured bytes versus analytic bytes, effective KV dtype, loaded
module precision/backend metadata, worker failures/version/rank gaps, immutable
revision checks, inherited tracing rejection, graph-file integrity/readability,
metadata redaction, and engine shutdown on extraction failure.
Additional record-extension checks preserve all existing ENG-72 fields and
Pamela's namespaced keys, separate canary/control instances, reject evidence
overwrites, avoid mutating the input, and never invent trial verdicts.

## Not yet verified

- Live Kimi checkpoint, container identity, engine version, baseline flags and
  resident weights. Rishith confirmed gitmachine-2 for this task on October 27.
- Installed engine's RPC compatibility, all-eight-rank visibility, and worker
  memory/quantization fields.
- Compiled graph dump production and shapes without GitM tracing hooks.
- Offline engine configuration parity with the team's HTTP serving deployment.
- Complete graph/rank/shape coverage, planner parity, and performance.

This is an initial R1 implementation. It does not complete ENG-85 acceptance,
replace ENG-69's interface, or implement ENG-86/87. The earlier DeepSeek job 131
does not validate this extractor or Kimi.
The `gitm.engine_descriptions` extension key is proposed for review following
Sebastian's October 27 direction to expand the existing ENG-72 run record.

## PR #217 CI corrections

The initial PR failed Ruff's UP038 rule in three `isinstance` calls: local Ruff
was 0.17.0, while CI pins 0.12.11. Use Python 3.10+ union syntax and verify with
the exact CI version. Repository-wide Ruff 0.12.11 now passes.

The full CI suite also exposed three probe-test failures when earlier tests left
`CUDA_INJECTION64_PATH` enabled. The extractor test fixture now isolates all
tracing environment variables. The dedicated tracing-rejection test still sets
an injection variable and verifies the production guard refuses it. All 37 new
tests pass even when the pytest process inherits an injection variable; no
production guard has been removed or bypassed.
