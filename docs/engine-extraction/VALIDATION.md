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

## Review pass — November 2, 2026

Changes addressing Sebastian's review, made on top of `089f5a6`:

- **Reuse across engines:** the engine-neutral helpers (`{status, value, source}`
  builders, JSON-safe serialization with secret redaction, the gap collector, and
  `extend_run_record`) moved to a new `gitm/engines/record.py`. `vllm_record.py`
  imports them and keeps only vLLM-specific config navigation. No behavior change;
  the vLLM module re-uses the same names. This is a small shared module, not a
  base class or a shared engine protocol (that is ENG-69/73's).
- **Wiring/scope:** `ENG-85.md` gains a "Wiring and boundaries" section stating
  exactly what the PR connects (only `extend_run_record` onto the ENG-72 record)
  and what it deliberately does not (the loop and Pamela's collector).
- **JSON organization:** `ENG-85.md` gains a "Record organization" rationale
  (three tiers: driver-resolved config, per-rank observations, diagnostics) and a
  complete, clearly-synthetic example nested inside an ENG-72 run record.
- **Acceptance checklist:** added `ACCEPTANCE-CHECKLIST.md`, every R1 field mapped
  to code, CPU test, and the live evidence still owed.

Re-run on this branch with Python 3.11:

```bash
PYTHONPATH=. python -m pytest tests/test_vllm_engine_record.py \
  tests/test_vllm_stats_v1.py tests/test_predict_graph_from_engine.py \
  tests/test_gpu_detect.py -q -o addopts=''
ruff --version   # 0.12.11
ruff check .
```

Result: **71 CPU tests pass** (unchanged — the refactor moved code, it did not add
or remove tests), `ruff check .` clean. No GPU was used.

## Live run blocker (code delivery)

The written ENG-85 acceptance needs one Kimi K2.6 run on the assigned node. That
run is currently **blocked**, and the blocker is not the extractor:

- The container installs gitm via `gitm-job --gitm-ref` (default `main`), which
  reads `GitM-Labs/runtime` — not a fork. The ENG-85 code lives on the fork
  branch `medha/eng-85-vllm-engine-record` and is **not merged to `main`**, so the
  default image does not contain it.
- Per Rishith's ENG-77 note, private-repo GitHub archive downloads inside
  containers now return **404**, and `--gitm-ref` installation of unmerged code
  needs a separately resolved private-repo delivery mechanism that is not yet in
  place. Injecting personal GitHub credentials into a job is not an acceptable
  workaround.

**Smallest unblock:** a supported way to get this branch's commit into the
container — e.g. the team merges the PR to `main` first (then `--gitm-ref main`
carries it), or Rishith's baked-in-image / private delivery path lands and we
point the job at this commit. Until then, the extractor is finished and green
locally but cannot be validated live.
