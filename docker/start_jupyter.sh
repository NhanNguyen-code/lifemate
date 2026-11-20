#!/usr/bin/env bash
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p /result
cd /code
jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser
