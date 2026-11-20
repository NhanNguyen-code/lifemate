"""Shims for transformers v4/v5 differences and GPUs without bf16."""
from __future__ import annotations

import re
import sys


def dtype_kw(dtype) -> dict:
    try:
        import transformers
        v = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    except Exception:
        v = (0, 0)
    return {"dtype": dtype} if v >= (4, 56) else {"torch_dtype": dtype}


def training_args(cls, warmup: float = 0.0, **kw):
    import inspect
    params = set(inspect.signature(cls.__init__).parameters)
    if "warmup_ratio" in params:      # v5 folded warmup_ratio into warmup_steps
        kw["warmup_ratio"] = warmup
    else:
        kw["warmup_steps"] = warmup
    if "eval_strategy" not in params and "eval_strategy" in kw:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    for k in dropped:
        kw.pop(k)
    if dropped:
        print(f"[compat] TrainingArguments: dropping options this transformers version lacks: {dropped}")
    return cls(**kw)


def gpu_capability() -> tuple[int, int] | None:
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return None


def auto_batch(per_item: int = 1) -> int:
    # a T4 holds the KV cache of ~4 prompts of 6-8k tokens; bigger batches evict each other's prefix cache
    try:
        import torch
        if torch.cuda.is_available():
            gib = torch.cuda.get_device_properties(0).total_memory / 2**30
            return (4 if gib < 20 else 8 if gib < 30 else 16) * per_item
    except Exception:
        pass
    return 16 * per_item


def bf16_ok() -> bool:
    cap = gpu_capability()
    return cap is not None and cap[0] >= 8


# without torchrun, Trainer wraps 2xT4 (Kaggle) in DataParallel; must run before torch is imported
def single_gpu():
    import os
    if "LOCAL_RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resolve_resume(out_dir, resume):
    if resume != "auto":
        return resume
    from pathlib import Path
    d = Path(out_dir)
    if not d.is_dir():
        return None
    ck = [p for p in d.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    if not ck:
        return None
    last = max(ck, key=lambda p: int(p.name.split("-")[-1]))
    print(f"[resume] from {last}")
    return str(last)


def hide_old_torchao(min_version: str = "0.16.0"):
    # Colab ships torchao 0.10 and recent peft raises on it instead of skipping it; nothing here uses torchao
    try:
        from importlib.metadata import version
        from packaging.version import Version
        if "torchao" not in sys.modules and Version(version("torchao")) < Version(min_version):
            sys.modules["torchao"] = None
    except (ImportError, ValueError):
        pass
