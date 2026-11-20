# LifeMate image, laid out as in BTC's docker guide (ZAC 2026):
#   /code/predict.sh                  reads /data/<split>/, writes /result/submission.jsonl
#   /code/start_jupyter.sh            timing notebook (port 9777) -> /result/jupyter_submission.jsonl
#   /code/saved_models/{llm,asr}      merged fine-tuned weights only, no base models
#   /code/saved_models/asr2           gipformer (untuned) when assets/fuse.json turns fusion on
#   /code/training_code/              code + README to retrain; data/ = synthetic training data (teacher thoughts)
# Build from the repo root once models/llm, models/asr and assets/ are filled in:
#   docker build -f docker/Dockerfile -t zac2026-teamname:v1 .
# vllm-openai v0.19.0 ships CUDA 12.9 + torch 2.10: under BTC's CUDA 13.2 cap and built for sm_120 (RTX 5090).
ARG VLLM_TAG=v0.19.0
FROM vllm/vllm-openai:${VLLM_TAG}

ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
WORKDIR /code

COPY requirements-docker.txt /code/requirements.txt
# fail the build if pip swapped out the torch that vllm was built against
RUN python3 -c "import torch; print(torch.__version__)" > /tmp/torch_version \
 && python3 -m pip install --no-cache-dir -r /code/requirements.txt \
 && python3 -c "import torch, vllm; v = open('/tmp/torch_version').read().strip(); \
assert torch.__version__ == v, (v, torch.__version__); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)" \
 && rm /tmp/torch_version

# weights before code so code edits don't re-copy ~11 GB
COPY models/llm /code/saved_models/llm
COPY models/asr /code/saved_models/asr
# second ASR (gipformer, 68M) for fusion of the final request; holds only a README when fusion is off
COPY models/asr2 /code/saved_models/asr2
COPY assets /code/assets
COPY configs /code/configs
COPY lifemate /code/lifemate
COPY docker/predict.sh docker/predict.py docker/preprocessing.py docker/start_jupyter.sh \
     docker/predict_notebook.ipynb /code/

COPY lifemate /code/training_code/lifemate
COPY scripts /code/training_code/scripts
COPY configs /code/training_code/configs
COPY colab /code/training_code/colab
COPY requirements.txt requirements-gpu.txt /code/training_code/
COPY docker/TRAINING.md /code/training_code/README.md
COPY training_data /code/training_code/data

RUN chmod +x /code/predict.sh /code/start_jupyter.sh
EXPOSE 9777
# the base image's entrypoint is `vllm serve`; BTC passes `/bin/bash /code/predict.sh` as the command
ENTRYPOINT []
CMD ["/bin/bash", "/code/predict.sh"]
