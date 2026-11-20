"""Weighted sum of two LoRA adapters of the same base model, as one adapter (no training).

A LoRA adds s * B @ A to a weight matrix (s = lora_alpha / r). Stacking A1 over A2 and putting lam * B1 next to
(1 - lam) * B2 gives an adapter of rank r1 + r2 that adds exactly lam * s1 * B1 @ A1 + (1 - lam) * s2 * B2 @ A2:
vLLM serves it as it is, and merge-lora turns it into one model of the base model's size.

    python -m lifemate soup-lora --a checkpoints/sft-v5 --b checkpoints/sft-v3 --lam 0.5 --thoughts off \
        --out checkpoints/sft-soup
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
TOKENIZER = ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json", "merges.txt",
             "added_tokens.json", "chat_template.jinja")


def _scale(cfg: dict, r: int | None = None) -> float:
    r = int(cfg["r"]) if r is None else r
    alpha = float(cfg.get("lora_alpha", cfg["r"]))
    return alpha / r ** 0.5 if cfg.get("use_rslora") else alpha / r


def _alpha_for(cfg: dict, scale: float, r: int) -> float:
    return scale * r ** 0.5 if cfg.get("use_rslora") else scale * r


def soup(a: str, b: str, lam: float, out: str, thoughts: str = "a") -> dict:
    """Write the adapter lam * a + (1 - lam) * b to `out`. thoughts: prompt format of the result, taken from adapter
    "a" or "b", or "on" / "off"."""
    import torch
    from safetensors.torch import load_file, save_file
    from .prompt import find_model_prompt, write_model_prompt
    if not 0.0 <= lam <= 1.0:
        raise SystemExit(f"--lam must be in [0, 1], got {lam}")
    ca, cb = (json.loads((Path(p) / CONFIG).read_text(encoding="utf-8")) for p in (a, b))
    for key in ("rank_pattern", "alpha_pattern"):
        if ca.get(key) or cb.get(key):
            raise SystemExit(f"adapters with {key} are not supported")
    if set(ca.get("target_modules") or []) != set(cb.get("target_modules") or []):
        raise SystemExit("the two adapters target different modules")
    na, nb = (Path(str(c.get("base_model_name_or_path") or "")).name for c in (ca, cb))
    if na and nb and na != nb:
        print(f"[soup] warning: different base models in the configs ({na} / {nb})", flush=True)
    wa, wb = load_file(str(Path(a) / WEIGHTS)), load_file(str(Path(b) / WEIGHTS))
    if set(wa) != set(wb):
        raise SystemExit("the two adapters hold different tensors")
    sa, sb = _scale(ca), _scale(cb)
    r = int(ca["r"]) + int(cb["r"])
    merged = {}
    for k in sorted(wa):
        x, y = wa[k].float(), wb[k].float()
        if k.endswith("lora_A.weight"):
            merged[k] = torch.cat([x, y], dim=0).contiguous()
        elif k.endswith("lora_B.weight"):                    # the result keeps adapter a's scale
            merged[k] = torch.cat([lam * x, (1.0 - lam) * (sb / sa) * y], dim=1).contiguous()
        else:
            raise SystemExit(f"unsupported tensor in an adapter: {k}")
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(o / WEIGHTS), metadata={"format": "pt"})
    cfg = dict(ca)
    cfg.update(r=r, lora_alpha=_alpha_for(ca, sa, r))
    (o / CONFIG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for f in TOKENIZER:
        if (Path(a) / f).is_file():
            shutil.copy(Path(a) / f, o / f)
    if thoughts in ("a", "b"):
        src = find_model_prompt(a if thoughts == "a" else b)
        th = bool(json.loads(src.read_text(encoding="utf-8")).get("thoughts")) if src else False
    else:
        th = thoughts == "on"
    write_model_prompt(o, {"thoughts": th})
    info = {"a": str(a), "b": str(b), "lam": lam, "rank": r, "thoughts": th}
    (o / "soup.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate soup-lora")
    ap.add_argument("--a", required=True, help="adapter weighted by --lam")
    ap.add_argument("--b", required=True, help="adapter weighted by 1 - lam (same base model, same target modules)")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--thoughts", choices=["a", "b", "on", "off"], default="a",
                    help="prompt format of the result: like adapter a / b, or thoughts on / off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    info = soup(a.a, a.b, a.lam, a.out, a.thoughts)
    print(f"[soup] {info} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
