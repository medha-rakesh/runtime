# ENG-85: vLLM engine record

## Scope and implementation plan

Start from runtime main `bbc6c07587cece4dfa42ef6a282bcb253fc2a942`.
Sebastian confirmed on October 27 that development can start from main. In his
subsequent clarification he linked ENG-72's Run record section and asked that it
be expanded for the values this task needs. The JSON therefore supplies an
engine-description extension to that record, preserving its existing verdict
and provenance fields. It does not introduce a replacement loop record.

1. Check the engine's compilation dump independently of GitM's tracing hooks.
2. Read vLLM's resolved configuration, preserving sources and missing values.
3. Read worker-local model precision, memory measurements and module information
   through vLLM's worker RPC. Keep each rank separate; never substitute whole-node
   NVML usage or a hand-derived parameter count for measured model memory.
4. Describe compilation artifacts by path and content hash; a file's existence
   alone does not prove graph completeness or tensor-shape coverage.
5. Supply an opt-in launch probe and CPU tests. The Kimi K2.6 acceptance check
   remains pending until the team supplies the checkpoint, image/version,
   launch settings and node availability.

The extractor will be a small function in `gitm.engines.vllm_record`, callable
from an adapter when ENG-69/ENG-73's shared interface lands. Do not invent that
interface or change existing planner dispatch in this R1 change.

## Evidence rules

Each extracted field has a JSON-safe value, a source, and an explicit status.
Missing values remain missing; missing fields are listed as gaps. Do not read
Hugging Face files independently, infer precision from a model name, derive
memory from shapes, or report requested launch arguments as resolved state.

Capture engine version and observation time. Worker data must identify its rank.
Configuration-only results do not establish post-initialization memory truth.
Do not disable compilation to pass the first check or enable GitM/CUPTI/NVTX
hooks for this extraction run.

## Acceptance and validation

- CPU tests: config layouts, unavailable values, source attribution, JSON output,
  per-rank results and artifact integrity. They do not validate a live vLLM build.
- GPU check: exact Kimi K2.6 checkpoint and engine/image pinned, eight H200s on
  the assigned node, compiled graph artifacts readable without GitM hooks, and
  every required category populated or reported as an explicit gap.
- Reviewer: Sebastian; planner reviewer: Adit. Moh owns the Kimi baseline and
  architecture checker; Rishith owns queue/node setup; Collin owns the shared
  engine interface. Confirm runtime details with them before scheduling a run.

No GPU run has been performed for ENG-85 yet.

## What this change supplies

- `extract_vllm_record(llm, graph_dir=...)`: describe an already initialized
  in-process `vllm.LLM` or `LLMEngine`. No engine is started by this function.
- `inspect_worker`: vLLM's `collective_rpc` callable for rank-local profiling,
  effective KV dtype, loaded model class, applied quantization methods and
  parameter/buffer shapes and storage dtypes. It copies no tensor data.
- A command-line launch probe using vLLM's own `EngineArgs` parser and
  `LLM.from_engine_args`. It exercises one generated token and shuts the engine
  down after recording. It refuses a version mismatch, moving revision, injected
  tracer, disabled compilation, or an existing output directory before launch.
- `deploy/h200/eng85-extract.sh`: an opt-in wrapper for an allocated GPU job.
- `gitm.engines.record`: engine-neutral helpers — the `{status, value, source}`
  field builders, the strict JSON-safe metadata conversion with secret
  redaction, the gap collector, and `extend_run_record`. `vllm_record` imports
  these and keeps only vLLM-specific config navigation, so a later SGLang
  extractor reuses the serialization and the ENG-72 extension rule instead of
  restating them. This is a small set of functions, not a base class or a shared
  engine protocol — that protocol is ENG-69/ENG-73's to define.

Existing integration points remain unchanged:

| Existing code | Reuse / relationship |
| --- | --- |
| `gitm/tracer/vllm_stats.py::engine_config_value` | Existing pattern for resolved config roots; extractor also records the successful source path. |
| `gitm/planner/context.py::build_planner_context` | Current planner deployment consumer; wiring the new record belongs to R2. |
| `gitm/scheduler/loop.py::_execution_graph_basis` | Still builds family graphs; this R1 change does not replace it with an FX graph (R3). |
| `gitm/serve/artifacts.py` | Capture manifest describes collection; it is not this richer engine record. |

## Wiring and boundaries

This answers the review question of how the record connects to the loop and to
Pamela's collector. The short answer: in R1 it does not call either, by design.

**What this PR wires:**

- `extract_vllm_record(engine)` is a pure read of an already-initialized engine.
  It calls nothing in `gitm/scheduler/loop.py`, starts no engine, and submits no
  job. The one-token launch probe in `main` is the only code that constructs an
  engine, and only when a human runs the CLI.
- The single seam into existing code is `extend_run_record`, which attaches the
  engine record to ENG-72's run record under the proposed
  `gitm.engine_descriptions` key. It adds one namespaced key and changes no
  existing field. That is the whole integration surface in R1.

**What this PR deliberately does not wire, and why:**

- **The loop does not call the extractor.** `gitm/scheduler/loop.py` still builds
  family graphs through `_execution_graph_basis`; replacing that with engine
  truth is R2/R3 (ENG-86/87) and depends on the shared engine interface
  (ENG-69, Backlog) and the loop split (ENG-73, due Nov 6). The assignment says
  to consume that interface, not define it, so R1 stops at producing the record.
- **Pamela's collector is not modified.** Per ENG-83 the collector is
  telemetry-only: it does not own engine lifecycle and does not write verdicts.
  This PR does not touch its code or its tick path. The engine record is a
  separate, static description that *supplements* a run record; it is not a
  telemetry stream and is not emitted from the collector.

So the methods named in the table above are referenced to show where the record
will plug in later; this change does not edit them. When ENG-69/73 land, an
adapter behind that interface calls `extract_vllm_record` and the result flows
into the loop's run record through `extend_run_record` — no new path required
here.

## JSON structure

The CLI's `engine_record.json` is an **extension fragment**:

```json
{
  "gitm.engine_descriptions": {
    "<engine-instance-id>": {"schema_version": "gitm.engine-description.v1"}
  }
}
```

The abbreviated payload above omits evidence fields. `gitm.engine_descriptions`
is a **proposed extension key for Sebastian/Pamela review**, not a pre-existing
approved database column. It follows the design's `gitm.*` namespace rule.
`extend_run_record(existing, description, instance_id=...)` adds it to the
existing record without changing any existing fields. Multiple entries preserve
control and canary settings separately; different evidence for the same instance
is refused rather than silently overwritten. The caller owns the association
with ENG-72's canary/control instance IDs. CLI observations use a new `probe-*`
label unless the caller provides `--instance-id`; it is not a Slurm job ID,
production service identity, or loop run ID.

| ENG-72 existing fields | Treatment |
| --- | --- |
| `beat_baseline`, `delta`, `iterations`, `wall_time_s`, `node_healthy_for_window` | Preserve when supplied by the loop; never synthesize from extraction. |
| `lever_id`, `knob_values`, `canary_instance`, `control_instances`, `window_ids` | Preserve caller-provided context. |
| `noise_floor`, `threshold`, `min_effect`, `slo_guard`, `state`, `policy_id`, `degradations` | Preserve existing decisions, policy and uncertainty. |
| `baseline_config`, `candidate_config` | Preserve the complete launch-argument snapshots; resolved engine evidence supplements them. |
| New `gitm.engine_descriptions[instance_id]` | Model, precision, deployment, worker memory, graph sources and extraction gaps. |

Pamela's namespaced equivalents (`gitm.beat_baseline`, etc.) are also preserved
unchanged. With no existing loop record, the helper returns only the extension;
that is not a valid completed performance trial and cannot prove a healthy run
or a gain. CLI stdout labels it as a run-record engine extension.

Within each engine payload, `model`, `static_config`,
`deployment`, `compilation`, and `speculative` describe driver-side resolved
objects. `workers` preserves each rank's own observations. A field is
`{status, value, source}`; unavailable fields also have a reason. Explicit `null`
(e.g. speculation disabled) is distinct from an unavailable field.

`graph_artifacts` indexes readable debug text with byte lengths and SHA-256
hashes. A graph-shaped file is a candidate, **not** proof of complete FX coverage,
tensor-shape coverage, rank coverage, or execution order. Module enumeration is
a static tree, not an execution trace. The JSON always says live acceptance is
pending; no graph-to-planner conversion is claimed.

The linked base design is [ENG-72: Run record](https://linear.app/git-machine/document/loop-design-schema-otel-core-keys-vendor-extension-namespaces-a7eee4e7d4cd#heading-run-record-39dee12a).

`gaps` includes missing fields, RPC failures, incomplete rank coverage, worker
version mismatch and absent profiling measurements. Storage dtype in tensor
metadata does not establish an operation's arithmetic/activation dtype. Preserve
raw quant-method classes and engine exclusions instead of guessing that recipe.

### Record organization

The review asked why `model`, `static_config`, `deployment`, `compilation`,
`graph_artifacts` and `gaps` sit where they do. The payload is organized in three
tiers, and the ordering is deliberate:

1. **Driver-resolved configuration** — `model`, `static_config`, `deployment`
   (`parallel` / `cache` / `scheduler` / `attention`), `compilation`,
   `speculative`. These are what vLLM resolved on the driver process: identity,
   shape, precision intent and deployment settings. They come from one place
   (the resolved `vllm_config`) and are true for the whole engine.
2. **Per-rank observations** — `workers[]`. Anything that can differ by rank, or
   that only exists after the profile run, lives here: measured memory, effective
   KV dtype, `num_gpu_blocks`, loaded module/quant classes and tensor metadata.
   It is a list, not a merge, because a disagreeing rank is evidence, not noise.
3. **Diagnostics and provenance** — `graph_artifacts`, `gaps`, `status`,
   `acceptance`, plus top-level `engine` / `engine_version` / `observed_at`.

`graph_artifacts` is kept **out** of the core configuration tiers on purpose: it
is a content-hash index of readable debug files, not resolved engine state, and
its `graph_completeness` / `shape_coverage` / `rank_coverage` are always
`unverified` in R1. Folding it into `compilation` would blur a proven setting
(compile mode) with an unproven artifact index. `gaps` is a flat, top-level
digest so a reviewer can see every unavailable or partial field in one list
without walking the tree; each entry names the dotted path back into the record.

### Complete example (synthetic)

**This is a hand-written illustration, not live Kimi output.** Every value is a
placeholder (revisions shown as `aaaa…`, memory as round numbers), and repeated
fields are shown once per group with the rest elided. It shows the full nesting:
the engine record lives under one instance id inside an ENG-72 run record.

```json
{
  "schema": "gitm.run-record.v1",
  "run_id": "SYNTHETIC-not-a-real-run",
  "beat_baseline": false,
  "canary_instance": "cand-synthetic",
  "control_instances": ["base-synthetic"],
  "baseline_config": {"tensor_parallel_size": 8, "max_num_seqs": 64},
  "gitm.engine_descriptions": {
    "cand-synthetic": {
      "schema_version": "gitm.engine-description.v1",
      "engine": "vllm",
      "engine_version": {"status": "observed", "value": "0.31.0",
                         "source": "importlib.metadata.version('vllm')"},
      "observed_at": "2026-01-01T00:00:00+00:00",
      "model": {
        "model": {"status": "observed", "value": "moonshotai/Kimi-K2.6-SYNTHETIC",
                  "source": "llm_engine.vllm_config.model_config.model"},
        "revision": {"status": "observed", "value": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                     "source": "llm_engine.vllm_config.model_config.revision"},
        "quantization": {"status": "observed", "value": "fp8",
                         "source": "llm_engine.vllm_config.model_config.quantization"},
        "resolved_config_commit": {"status": "observed", "value": "aaaaaaaa...",
                                   "source": "...model_config.hf_config._commit_hash"}
      },
      "static_config": {"status": "observed",
                        "value": {"hidden_size": 7168, "n_routed_experts": 384},
                        "source": "...model_config.hf_text_config.to_dict()"},
      "deployment": {
        "parallel": {"tensor_parallel_size": {"status": "observed", "value": 8,
                     "source": "...parallel_config.tensor_parallel_size"}},
        "cache": {"num_gpu_blocks": {"status": "observed", "value": 12345,
                  "source": "...cache_config.num_gpu_blocks"}},
        "scheduler": {"max_num_seqs": {"status": "observed", "value": 64,
                      "source": "...scheduler_config.max_num_seqs"}},
        "attention": {"backend": {"status": "unavailable", "value": null,
                      "source": "...attention_config.backend",
                      "reason": "not exposed by this engine object"}}
      },
      "compilation": {"mode": {"status": "observed", "value": 3,
                      "source": "...compilation_config.mode"}},
      "speculative": {"status": "observed", "value": null,
                      "source": "llm_engine.vllm_config.speculative_config"},
      "graph_artifacts": {
        "status": "observed",
        "source": "/out/SYNTHETIC/compilation",
        "value": [{"path": "rank_0/computation_graph.py", "bytes": 4096,
                   "sha256": "0000...", "graph_candidate": true}],
        "errors": [],
        "graph_candidates": 1,
        "graph_completeness": "unverified",
        "shape_coverage": "unverified",
        "rank_coverage": "unverified"
      },
      "workers": [
        {
          "rank": {"status": "observed", "value": 0, "source": "worker.rank"},
          "device": {"status": "observed", "value": "cuda:0", "source": "worker.device"},
          "memory_bytes": {
            "model_memory_usage": {"status": "observed", "value": 90000000000,
                                   "source": "worker.model_runner.model_memory_usage"}
          },
          "effective_kv_dtype": {"status": "observed", "value": "torch.float8_e4m3fn",
                                 "source": "worker.model_runner.kv_cache_dtype"},
          "precision": {"class": {"status": "observed",
                        "value": "vllm....Fp8Config",
                        "source": "worker.get_model().quant_config.__class__"}},
          "modules": {"status": "observed", "value": [], "source": "...named_modules()"}
        }
      ],
      "gaps": [
        {"field": "deployment.attention.backend", "reason": "not exposed by this engine object"}
      ],
      "status": "partial",
      "acceptance": "pending_live_review"
    }
  }
}
```

The run-record keys (`beat_baseline`, `canary_instance`, `baseline_config`, …)
are ENG-72's and are preserved untouched; only `gitm.engine_descriptions` is
added. With no base record the helper returns just that one key — an extension
fragment, never a completed trial.

## Source compatibility checked

API inspection is against official vLLM **v0.31.0** source, not a live installation:

- [LLM.from_engine_args / collective_rpc](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/llm.py)
- [EngineArgs CLI parsing](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)
- [GPUWorker profiling and num_gpu_blocks](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_worker.py)
- [GPUModelRunner.model_memory_usage / effective KV dtype](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_model_runner.py)
- [CompilationConfig.debug_dump_path](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/compilation.py)

`peak_activation_memory` in that engine includes an applied CUDA-graph memory
estimate; it is not an isolated tensor-activation peak. Available KV memory is
a profiling budget, not measured live cache occupancy. `num_gpu_blocks` is read
from initialized worker state; no `blocks * block_size` capacity is invented.

The probe starts an **offline in-process LLM**, not an HTTP server. EngineArgs
must match the approved serving configuration, but HTTP-server deployment parity
is unverified until reviewed. Existing HTTP endpoints are not assumed to expose
worker profiling or the resolved quantization recipe. Import availability,
worker RPC, compiled execution and actual dump production must be checked on
the team's installed build; passing CPU tests does not settle these.

## First GPU check (template, not a submitted job)

Target **gitmachine-2**, confirmed by Rishith to Medha on October 27 despite
ENG-85's original node-1 wording. Get from the node/baseline evidence: exact
checkpoint SHA, image/version, baseline engine flags and resident weights.
Confirm the installed health checker and results retrieval.
Use the team's job-submission path and install this review branch/commit inside
the container first. `gitm-submit --gitm-ref` reads GitM-Labs/runtime, not a fork;
do not point it at a fork-only branch and expect installation to work.

Inside the allocated container, with the four required environment variables set:

```bash
bash deploy/h200/eng85-extract.sh <the team's approved vLLM engine flags>
```

Required variables: `MODEL`, `MODEL_REVISION` (40-character SHA), `VLLM_VERSION`
(exact installed version), `ENG85_OUT` (fresh results subdirectory). The flags
are EngineArgs (`--model`/`--revision` supplied by the wrapper), not HTTP server
options such as `--host` or `--port`. Do not copy shell placeholders literally.
No `gitm capture`, CUPTI injection or layerwise NVTX flag is used. The probe
refuses inherited tracing variables instead of silently calling that untraced.

Inspect `engine_record.json`, `compilation/`, and the job log. A zero probe exit
means graph candidates were indexed, not that all fields or ranks were complete.
Review `gaps`, verify all eight ranks, confirm shapes/graph readability and the
deployment match. File engine-layer gaps for unavailable required fields. Do not
remove catalogue entries or claim planner parity in this first check.
