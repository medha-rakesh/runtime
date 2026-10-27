#!/bin/bash
# Run INSIDE an allocated job with this branch installed. This does not submit a job.
set -euo pipefail
: "${MODEL:?Set MODEL to the team-confirmed Kimi model ID or local path}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the team-confirmed checkpoint SHA}"
: "${VLLM_VERSION:?Set VLLM_VERSION to the exact installed engine version}"
: "${ENG85_OUT:?Set ENG85_OUT to a fresh directory under the job results mount}"
if [ "$#" -eq 0 ]; then
    echo "Pass the team's vLLM engine flags, including parallelism and context limits." >&2
    exit 2
fi
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/out/hf_modules}"
exec python3 -m gitm.engines.vllm_record \
    --out "$ENG85_OUT" --expected-vllm-version "$VLLM_VERSION" \
    -- --model "$MODEL" --revision "$MODEL_REVISION" "$@"
