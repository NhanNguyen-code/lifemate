"""Merge a PhoWhisper LoRA into a full model dir (models/asr for Docker)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate merge-whisper")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from ..compat import dtype_kw
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    m = WhisperForConditionalGeneration.from_pretrained(a.base, **dtype_kw(dt))
    m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
    m.save_pretrained(a.out, safe_serialization=True)
    src = a.adapter if (Path(a.adapter) / "preprocessor_config.json").exists() else a.base
    WhisperProcessor.from_pretrained(src).save_pretrained(a.out)
    from .context import SETTINGS_FILE
    if (Path(a.adapter) / SETTINGS_FILE).exists():   # decoding settings (prompt, beams) travel with the weights
        (Path(a.out) / SETTINGS_FILE).write_bytes((Path(a.adapter) / SETTINGS_FILE).read_bytes())
    print(f"merged whisper written to {a.out}")


if __name__ == "__main__":
    main()
