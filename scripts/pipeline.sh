#!/usr/bin/env bash
# Pipeline stages.
# env: OUT BASE GIP_DIR WHISPER_BASE PY TAG TRAIN_ARGS INFER_ARGS PREDICT_ARGS TRANSCRIPTS BIAS TEAM IMAGE
set -euo pipefail
DATA="${DATA:?set DATA to the dir holding train/ and public_test/}"
OUT="${OUT:-outputs}"
BASE="${BASE:-models/base-llm}"
GIP_DIR="${GIP_DIR:-models/gipformer}"
WHISPER_BASE="${WHISPER_BASE:-models/phowhisper-large}"
PY="${PY:-python}"
TAG="${TAG:-v1}"
SPLIT_FILE="$OUT/split_apps.json"
VAL="$OUT/split_apps.val_ids.txt"
mkdir -p "$OUT"
L="$PY -m lifemate"
TEAM="${TEAM:-teamname}"
IMAGE="${IMAGE:-zac2026-$TEAM:v1}"

case "${1:-help}" in
  day1)   # CPU only
    $L experiments --data "$DATA" --out "$OUT/experiments.json"
    $L make-split --data "$DATA" --holdout apps --n-apps 10 --out "$SPLIT_FILE"
    $L make-split --data "$DATA" --holdout random --n-val 200 --out "$OUT/split_random.json"
    $L build-assets --data "$DATA" --out assets
    ;;
  submit-empty)   # format check for the portal; scores about the out-of-scope share
    $L infer --data "$DATA" --split public_test --backend empty --text oracle --out "$OUT/submission_empty.jsonl"
    $L validate --data "$DATA" --split public_test --submission "$OUT/submission_empty.jsonl"
    ;;
  asr-untuned)   # untuned gipformer on train, used as SFT input text
    $L transcribe --data "$DATA" --split train --asr gipformer --model "$GIP_DIR" --out "$OUT/asr_train_untuned.jsonl"
    $L asr-eval --data "$DATA" --split train --transcripts "$OUT/asr_train_untuned.jsonl"
    ;;
  asr-ft-dev)   # PhoWhisper LoRA without val audio, for end-to-end dev numbers
    # ASR_ARGS: extra finetune-whisper flags
    $L finetune-whisper --data "$DATA" --model "$WHISPER_BASE" --val-ids "$VAL" --out checkpoints/phowhisper-lora-dev \
        ${ASR_ARGS:-}
    $L transcribe --data "$DATA" --split train --ids-file "$VAL" --asr whisper --model "$WHISPER_BASE" \
        --lora checkpoints/phowhisper-lora-dev --out "$OUT/asr_val_ft.jsonl"
    $L asr-eval --data "$DATA" --split train --ids-file "$VAL" --transcripts "$OUT/asr_val_ft.jsonl"
    ;;
  asr-ft-full)   # final ASR: all train audio, merged into models/asr
    # ASR_DECODE: decoding settings stored next to the weights
    $L finetune-whisper --data "$DATA" --model "$WHISPER_BASE" --out checkpoints/phowhisper-lora-full ${ASR_ARGS:-}
    if [ -n "${ASR_DECODE:-}" ]; then
        $L asr-settings --dir checkpoints/phowhisper-lora-full $ASR_DECODE
    fi
    $L merge-whisper --base "$WHISPER_BASE" --adapter checkpoints/phowhisper-lora-full --out models/asr
    ;;
  baseline)   # base model + few-shot, gold text, leave-apps-out val
    $L infer --data "$DATA" --split train --ids-file "$VAL" --backend vllm --model "$BASE" --text oracle \
        --out "$OUT/base_val.jsonl" --trace "$OUT/base_val.trace.jsonl" ${INFER_ARGS:-}
    $L score --data "$DATA" --split train --ids-file "$VAL" --submission "$OUT/base_val.jsonl" --report "$OUT/base_val.report.json"
    ;;
  sft)
    $L build-sft --data "$DATA" --split-file "$SPLIT_FILE" --transcripts "$OUT/asr_train_untuned.jsonl" \
        --out "$OUT/sft_$TAG.jsonl"
    $L train --model "$BASE" --data "$OUT/sft_$TAG.jsonl" --out "checkpoints/sft-$TAG" ${TRAIN_ARGS:-}
    $L infer --data "$DATA" --split train --ids-file "$VAL" --backend vllm --model "$BASE" --lora "checkpoints/sft-$TAG" \
        --text oracle --out "$OUT/sft_${TAG}_val_oracle.jsonl" --trace "$OUT/sft_${TAG}_val_oracle.trace.jsonl" ${INFER_ARGS:-}
    $L score --data "$DATA" --split train --ids-file "$VAL" --submission "$OUT/sft_${TAG}_val_oracle.jsonl" \
        --report "$OUT/sft_${TAG}_val_oracle.report.json"
    ;;
  eval-asr)   # same adapter on ASR text; the gap to `sft` is what ASR costs
    TR="${TRANSCRIPTS:-$OUT/asr_val_ft.jsonl}"
    $L infer --data "$DATA" --split train --ids-file "$VAL" --backend vllm --model "$BASE" --lora "checkpoints/sft-$TAG" \
        --text asr --transcripts "$TR" --out "$OUT/sft_${TAG}_val_asr.jsonl" --trace "$OUT/sft_${TAG}_val_asr.trace.jsonl" ${INFER_ARGS:-}
    $L score --data "$DATA" --split train --ids-file "$VAL" --submission "$OUT/sft_${TAG}_val_asr.jsonl" \
        --report "$OUT/sft_${TAG}_val_asr.report.json"
    ;;
  rft)
    $L rft --data "$DATA" --split-file "$SPLIT_FILE" --model "$BASE" --lora "checkpoints/sft-$TAG" \
        --transcripts "$OUT/asr_train_untuned.jsonl" --k 4 --out "$OUT/rft_$TAG.jsonl"
    $L train --model "$BASE" --data "$OUT/sft_$TAG.jsonl" "$OUT/rft_$TAG.jsonl" --out "checkpoints/sft-$TAG-rft" ${TRAIN_ARGS:-}
    ;;
  calibrate)
    $L calibrate --data "$DATA" --ids-file "$VAL" --model "$BASE" --lora "checkpoints/sft-${CAL_TAG:-$TAG}" \
        --text asr --transcripts "${TRANSCRIPTS:-$OUT/asr_val_ft.jsonl}" --out assets/bias.json
    ;;
  thoughts)   # a teacher LLM writes a short reason for every gold step of train (SFT data only, not in the image)
    $PY scripts/download_models.py --dir models "${TEACHER:-teacher}"     # teacher | teacher-small
    $L rationale --data "$DATA" --model "models/${TEACHER:-teacher}" --out "$OUT/thoughts_train.jsonl"
    ;;
  synth)   # new requests on train states, solved twice by the teacher; kept when both runs agree (development only)
    # TEACHER=teacher35 (Qwen3.5-35B-A3B-FP8, the one in the image's data) or teacher (Qwen3-30B-A3B, failed the check)
    T="${TEACHER:-teacher35}"; TAG="${SYN_TAG:-q35}"
    $PY scripts/download_models.py --dir models "$T"
    $L synth --data "$DATA" --model "models/$T" --k 5 --out "$OUT/synth_train_$TAG.jsonl" \
        --thoughts-out "$OUT/synth_thoughts_$TAG.jsonl" --check-ids "$OUT/split_apps.val_ids.txt" \
        --report "$OUT/synth_check_$TAG.json" --hash-cache "$OUT/gold_hash_cache.json"
    ;;
  final-v5)   # v5 recipe on all of train: no thoughts, renamed apps 0.6, distractor apps 0.3, lr 1e-4
    # val samples get the dev PhoWhisper transcripts (that model never heard them), the rest untuned gipformer
    $L build-sft --data "$DATA" --transcripts "$OUT/asr_train_untuned.jsonl" "$OUT/asr_val_ft.jsonl" \
        --p-rename 0.6 --p-distractor 0.3 --out "$OUT/sft_v5.jsonl"
    $L train --model "$BASE" --data "$OUT/sft_v5.jsonl" --out checkpoints/sft-v5 --lr 1e-4 ${TRAIN_ARGS:-}
    ;;
  final-v3)   # v3 recipe on all of train: a teacher's thought before every action (`thoughts`), lr 2e-4
    TH=("$OUT/thoughts_train.jsonl")
    SYN=()
    TAG="${SYN_TAG:-q35}"
    if [ -n "${WITH_SYNTH:-}" ] && [ -f "$OUT/synth_train_$TAG.jsonl" ]; then      # v4 only: synthetic samples
        TH+=("$OUT/synth_thoughts_$TAG.jsonl"); SYN=(--extra-samples "$OUT/synth_train_$TAG.jsonl")
    fi
    $L build-sft --data "$DATA" --transcripts "$OUT/asr_train_untuned.jsonl" "$OUT/asr_val_ft.jsonl" \
        --thoughts "${TH[@]}" "${SYN[@]}" --out "$OUT/sft_v3.jsonl"
    $L train --model "$BASE" --data "$OUT/sft_v3.jsonl" ${RFT_FILE:+"$RFT_FILE"} --out checkpoints/sft-v3 \
        --lr 2e-4 ${TRAIN_ARGS:-}
    ;;
  final)   # FINAL_VARIANT (docker/TRAINING.md, 5): soup (the submission) | v5 | v5+v3 | v3 -> models/llm, assets/
    V="${FINAL_VARIANT:-soup}"
    case "$V" in
      v5|v3)
        bash "$0" "final-$V"
        $L merge-lora --base "$BASE" --adapter "checkpoints/sft-$V" --out models/llm
        ;;
      v5+v3)   # the base model with both adapters: v5 answers, v3 is asked at the first stop without a write
        bash "$0" final-v5
        bash "$0" final-v3
        rm -rf models/llm
        cp -r "$BASE" models/llm
        for pair in main:sft-v5 consult:sft-v3; do
            d="models/llm/adapters/${pair%%:*}"
            mkdir -p "$d"
            cp "checkpoints/${pair#*:}"/{adapter_config.json,adapter_model.safetensors,lifemate_prompt.json} "$d/"
        done
        ;;
      soup)   # one model: SOUP_LAM * v5 + (1 - SOUP_LAM) * v3, prompts as SOUP_THOUGHTS (on | off)
        bash "$0" final-v5
        bash "$0" final-v3
        $L soup-lora --a checkpoints/sft-v5 --b checkpoints/sft-v3 --lam "${SOUP_LAM:-0.5}" \
            --thoughts "${SOUP_THOUGHTS:-on}" --out checkpoints/sft-soup
        $L merge-lora --base "$BASE" --adapter checkpoints/sft-soup --out models/llm
        ;;
      *) echo "FINAL_VARIANT must be v5, v5+v3, soup or v3" >&2; exit 2 ;;
    esac
    $L build-assets --data "$DATA" --out assets
    ;;
  public)   # same models as the Docker image
    $L transcribe --data "$DATA" --split public_test --asr auto --model "${ASR_FINAL:-models/asr}" \
        --out "$OUT/asr_public.jsonl"
    $L infer --data "$DATA" --split public_test --backend vllm --model "${LLM:-models/llm}" --text asr \
        --transcripts "$OUT/asr_public.jsonl" --assets assets ${BIAS:+--bias "$BIAS"} --batch-size 1 \
        --out "$OUT/submission_public.jsonl" --trace "$OUT/public.trace.jsonl" ${INFER_ARGS:-}
    $L validate --data "$DATA" --split public_test --submission "$OUT/submission_public.jsonl"
    ;;
  public-predict)   # docker code path without docker (e.g. Colab/Kaggle GPU), per-sample time like BTC measures
    $L predict --data "$DATA" --split public_test --out "$OUT/predict_public.jsonl" --llm "${LLM:-models/llm}" \
        --asr-model "${ASR_FINAL:-models/asr}" --assets assets --config configs/docker.json \
        --bias "${BIAS:-assets/bias.json}" --trace "$OUT/predict_public.trace.jsonl" ${PREDICT_ARGS:-}
    $L validate --data "$DATA" --split public_test --submission "$OUT/predict_public.jsonl"
    ;;
  docker-build)
    # models/asr2: gipformer for fusion (assets/fuse.json); without it the folder only holds a README
    mkdir -p models/asr2
    if find models/asr2 -name "*.onnx" | grep -q .; then
        $L count-params --llm models/llm --asr models/asr --asr models/asr2
    else
        echo "no second ASR: fusion off" > models/asr2/README.txt
        $L count-params --llm models/llm --asr models/asr
    fi
    # synthetic training data goes into the image with the training code (rules: code + data)
    for f in "$OUT"/thoughts_train.jsonl "$OUT"/synth_train_*.jsonl "$OUT"/synth_thoughts_*.jsonl "$OUT"/synth_check_*.json; do
        if [ -f "$f" ]; then cp "$f" training_data/; fi
    done
    docker build -f docker/Dockerfile -t "$IMAGE" .
    ;;
  docker-test)   # BTC's run command plus --network none; DATA must hold public_test/
    mkdir -p result
    docker run --rm --gpus '"device=0"' --network none -e LIFEMATE_ARGS="${PREDICT_ARGS:-}" \
        -v "$(realpath "$DATA"):/data:ro" -v "$PWD/result:/result" "$IMAGE" /bin/bash /code/predict.sh
    $L validate --data "$DATA" --split public_test --submission result/submission.jsonl
    ;;
  docker-jupyter)   # http://localhost:9777 (zac2026) -> predict_notebook.ipynb -> Run All, then docker-check
    mkdir -p result
    docker run -it --rm --gpus '"device=0"' -p 9777:9777 -e LIFEMATE_ARGS="${PREDICT_ARGS:-}" \
        -v "$(realpath "$DATA"):/data:ro" -v "$PWD/result:/result" "$IMAGE" /bin/bash /code/start_jupyter.sh
    ;;
  docker-check)   # notebook rows must equal predict.sh rows, or BTC ignores the timing
    $L compare result/submission.jsonl result/jupyter_submission.jsonl
    ;;
  docker-save)   # upload the .tar.gz to Drive ("Anyone with the link"), submit link + md5
    docker save "$IMAGE" | gzip > "zac2026_$TEAM.tar.gz"
    md5sum "zac2026_$TEAM.tar.gz" | tee "zac2026_$TEAM.tar.gz.md5"
    ls -lh "zac2026_$TEAM.tar.gz"
    ;;
  *)
    sed -n '2,4p' "$0"
    echo "stages: day1 submit-empty asr-untuned asr-ft-dev asr-ft-full thoughts synth baseline sft eval-asr rft"
    echo "        calibrate final (final-v5 final-v3)"
    echo "        public public-predict"
    echo "docker: docker-build docker-test docker-jupyter docker-check docker-save"
    ;;
esac
