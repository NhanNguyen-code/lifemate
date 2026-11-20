"""Default settings of the inference run; command-line flags override them."""
import os
import shlex
import sys

sys.path.insert(0, "/code")

from lifemate.predict import PredictConfig, run  # noqa: E402

CFG = PredictConfig(
    data="/data",                          # BTC mounts the parent: /data/private_test/private_test.jsonl
    split=None,                            # private_test if present, else public_test, else the only split
    out="/result/submission.jsonl",
    backend="vllm",
    llm="/code/saved_models/llm",          # Qwen3-4B-Instruct-2507 + LoRA merged, or the base model + adapters/{main,consult}
    asr="auto",
    asr_model="/code/saved_models/asr",    # PhoWhisper-large + LoRA, merged
    asr_device="cuda",
    asr_beams=0,                           # 0 / "auto": beams and history prompt from saved_models/asr/lifemate_asr.json
    asr_prompt="auto",
    asr2_model="/code/saved_models/asr2",  # gipformer: second transcript of the final request, merged when
    fuse="auto",                           # assets/fuse.json exists (lifemate/asr/fuse.py)
    assets="/code/assets",                 # few-shot pool, value formats, bias.json
    config="/code/configs/docker.json",    # prompt / guard / agent settings of the submission
    bias="/code/assets/bias.json",
    gpu_mem=0.6,                           # vLLM share of the GPU (14 GB of an A30's 24, 19 of a 5090's 32); ASR sits beside it
    max_model_len=16384,
    seed=42,
)
CFG = CFG.override(shlex.split(os.environ.get("LIFEMATE_ARGS", "")))

if __name__ == "__main__":
    run(CFG.override(sys.argv[1:]))
