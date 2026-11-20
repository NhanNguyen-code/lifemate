"""Merge a LoRA adapter into its base model (for Docker, or vLLM without --lora)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate merge-lora")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    from .compat import dtype_kw
    model = AutoModelForCausalLM.from_pretrained(a.base, **dtype_kw(dt))
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    tok_src = a.adapter if (Path(a.adapter) / "tokenizer_config.json").exists() else a.base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(a.out)
    from .prompt import find_model_prompt, write_model_prompt
    f = find_model_prompt(a.adapter)
    if f is not None:
        write_model_prompt(a.out, json.loads(f.read_text(encoding="utf-8")))
        print(f"prompt settings {f.read_text(encoding='utf-8')} copied")
    print(f"merged model written to {a.out}")


if __name__ == "__main__":
    main()
