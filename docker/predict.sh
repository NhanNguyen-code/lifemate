#!/usr/bin/env bash
# Reads /data/<split>/, writes /result/submission.jsonl (id, tool_calls, state_hash, time).
set -euo pipefail
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p /result
cd /code
start=$(date +%s)
python3 /code/preprocessing.py "$@"
python3 /code/predict.py "$@"
echo "[time] predict.sh total: $(( $(date +%s) - start ))s"
