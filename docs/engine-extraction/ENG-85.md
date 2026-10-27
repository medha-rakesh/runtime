# ENG-85: vLLM engine record

## Scope and implementation plan

Start from runtime main `bbc6c07587cece4dfa42ef6a282bcb253fc2a942`.
Sebastian confirmed on October 27 that an initial JSON structure is acceptable
and that development can start from main. This record is an engine description,
not a loop performance verdict or a replacement database schema.

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

Existing integration points remain unchanged:

| Existing code | Reuse / relationship |
| --- | --- |
| `gitm/tracer/vllm_stats.py::engine_config_value` | Existing pattern for resolved config roots; extractor also records the successful source path. |
| `gitm/planner/context.py::build_planner_context` | Current planner deployment consumer; wiring the new record belongs to R2. |
| `gitm/scheduler/loop.py::_execution_graph_basis` | Still builds family graphs; this R1 change does not replace it with an FX graph (R3). |
| `gitm/serve/artifacts.py` | Capture manifest describes collection; it is not this richer engine record. |

## JSON structure

The initial schema is `gitm.engine-description.v1`. `model`, `static_config`,
`deployment`, `compilation`, and `speculative` describe driver-side resolved
objects. `workers` preserves each rank's own observations. A field is
`{status, value, source}`; unavailable fields also have a reason. Explicit `null`
(e.g. speculation disabled) is distinct from an unavailable field.

`graph_artifacts` indexes readable debug text with byte lengths and SHA-256
hashes. A graph-shaped file is a candidate, **not** proof of complete FX coverage,
tensor-shape coverage, rank coverage, or execution order. Module enumeration is
a static tree, not an execution trace. The JSON always says live acceptance is
pending; no graph-to-planner conversion is claimed.

`gaps` includes missing fields, RPC failures, incomplete rank coverage, worker
version mismatch and absent profiling measurements. Storage dtype in tensor
metadata does not establish an operation's arithmetic/activation dtype. Preserve
raw quant-method classes and engine exclusions instead of guessing that recipe.

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

Get from Moh/Rishith: exact checkpoint SHA; image/version; baseline engine flags;
resident weights and node (`gitmachine-1` per ENG-85; older infrastructure notes
put Kimi on node 2). Confirm the installed health checker and results retrieval.
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
