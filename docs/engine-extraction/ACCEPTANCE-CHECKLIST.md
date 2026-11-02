# ENG-85 / R1 acceptance checklist

Each R1 requirement mapped to the code that produces it, the CPU test that pins
its behavior, and the live GPU evidence it still needs. **No field below is
backed by a live Kimi K2.6 run yet** — the CPU column proves the extractor's
contract against fake engines; the "Live evidence" column is what a real run on
the assigned node must still supply. Do not read it as "done".

Legend: ✅ covered locally · ⏳ pending live GPU evidence · 📝 provenance-only.

| R1 requirement | Code | CPU test | Live evidence | Status |
| --- | --- | --- | --- | --- |
| **Model identity & static shape** (from vLLM's resolved objects, not HF) | `extract_vllm_record` → `_section(model_config)` + `_shape()` reads `hf_text_config/hf_config.to_dict()` | `test_engine_resolved_values_have_exact_sources_without_hf_reads`, `test_missing_config_never_becomes_a_default_model_or_precision` | Kimi identity + shape resolved by the installed engine | ✅ / ⏳ |
| **Precision recipe** incl. actual quantization class + per-module exclusions | `_quantization()` (class, `ignored_layers`, `modules_to_not_convert`, `weight_block_size`, `activation_scheme`, `is_checkpoint_fp8_serialized`); per-module `quant_method` in `_module_inventory` | `test_module_precision_distinguishes_applied_quant_method_and_tensor_dtype` | real resolved quant config + exclusions on the loaded Kimi model | ✅ / ⏳ |
| **Deployment — TP/PP/EP/DP** | `deployment.parallel` from `parallel_config` | source assertions in resolved-values test | live `world_size` = 8 and rank coverage | ✅ / ⏳ |
| **Deployment — cache + post-profile `num_gpu_blocks`** | `deployment.cache` + per-worker `cache.num_gpu_blocks` (read after init) | `test_memory_is_engine_profiled_and_rank_local_not_a_weight_formula` | `num_gpu_blocks` only exists after the live profile run | ⏳ |
| **Deployment — scheduler settings** | `deployment.scheduler` from `scheduler_config` | resolved-values test | live scheduler config | ✅ / ⏳ |
| **Deployment — attention backend** | per-module `get_attn_backend()` in `_module_inventory` | `test_resolved_attention_backend_is_read_from_the_loaded_module` | backend resolved by the installed build | ✅ / ⏳ |
| **Deployment — speculation** | `speculative` (explicit `null` ≠ missing) | resolved-values test asserts disabled→`null` | live speculative config | ✅ / ⏳ |
| **Engine memory — weight bytes** | `workers[].memory_bytes.model_memory_usage` (engine-profiled, not a shape formula) | `test_memory_is_engine_profiled...`, `test_missing_profile_is_not_replaced_by_nvml_or_zero` | real profiled bytes per rank | ⏳ |
| **Engine memory — activation peak** | `workers[].memory_bytes.peak_activation_memory` (raw field; includes CUDA-graph estimate in 0.31) | memory test | live profile | ⏳ |
| **Engine memory — KV capacity** | `workers[].memory_bytes.available_kv_cache_memory_bytes` + `kv_cache_config.num_blocks` | memory test | live profile budget | ⏳ |
| **Execution-graph source** — compile debug dump | `inspect_graph_artifacts()` indexes readable text with SHA-256; `graph_completeness/shape_coverage/rank_coverage = unverified` | `test_graph_index_checks_content_and_hashes...`, utf-8 chunk tests, `test_logs_alone_do_not_prove_a_graph_was_dumped` | dump actually produced by the installed build without GitM hooks | ✅ / ⏳ |
| **Execution-graph source** — module tree | `_module_inventory()` static tree + tensor metadata | `test_inventory_limit_is_explicit` | loaded module tree on the live model | ✅ / ⏳ |
| **Per-field source attribution** | every field is `{status, value, source}` via `gitm.engines.record.observe` | source asserted across tests | — | ✅ |
| **Engine version + observation time** | top-level `engine_version`, `observed_at`; per-worker version cross-check | `test_worker_version_mismatch_is_visible` | version of the installed build | ✅ / ⏳ |
| **No independent HF reads / no guessed values** | `_shape` uses only vLLM's resolved config; secret redaction; non-finite refusal; missing≠default | `test_nested_credentials_are_redacted`, `test_nonfinite_or_arbitrary_objects_are_not_silently_stringified`, `test_missing_config_never_becomes_a_default...` | — | ✅ |
| **First live check: compile dump readable WITHOUT GitM tracing hooks** | probe requires compilation enabled, refuses injected tracer / eager / unpinned revision / reused out dir | `test_probe_refuses_tracing_before_importing_vllm`, `test_probe_refuses_eager_or_unpinned_revision_before_build`, `test_probe_refuses_version_drift...` | dump readable on the installed build with no hooks | ✅ / ⏳ |
| **Run-record integration** (ENG-72, no fabricated verdicts) | `gitm.engines.record.extend_run_record` adds `gitm.engine_descriptions` only | 6 extension tests (preserve fields, separate instances, refuse overwrite, no fabricated verdict) | — | ✅ |
| **Rank coverage — all 8 ranks** | `workers` gap checks vs `world_size` | `test_missing_duplicate_or_wrong_ranks_block_complete_extraction` | all 8 H200 ranks present and consistent | ⏳ |
| **HTTP-serving parity** | probe is in-process offline LLM; `http_serving_parity = unverified` | — | compare probe config vs the live HTTP deployment | ⏳ 📝 |

## Node criterion (do not conflate)

- **Written acceptance (Linear ENG-85):** Kimi K2.6 on **gitmachine-1**.
- **Rishith, Oct 27:** use **gitmachine-2** for this task.
- A gitmachine-2 probe is useful evidence but **does not** satisfy the written
  node-1 criterion. If the run happens on node 2, record it as node-2 evidence
  and flag the node-1 requirement as still open.

## Overall

Local contract: **complete and green** (71 CPU tests, ruff 0.12.11 clean).
Live acceptance: **not met** — every ⏳ row needs one bounded Kimi K2.6 run on
the assigned node, which is currently blocked on code delivery into the
container (see `VALIDATION.md` → "Live run blocker"). ENG-85 is **not complete**
until that run populates the ⏳ rows or files each as an explicit engine-layer gap.
