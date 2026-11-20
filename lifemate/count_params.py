"""Count parameters from checkpoint headers without loading the models. Limits: LLM < 5B, total < 6B."""
from __future__ import annotations

import argparse
import json
import struct
from math import prod
from pathlib import Path


def safetensors_numel(path: Path) -> int:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sum(prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def onnx_numel(path: Path) -> int:
    try:
        import onnx
        m = onnx.load(str(path), load_external_data=False)
        return sum(prod(t.dims) for t in m.graph.initializer)
    except Exception:
        size = path.stat().st_size
        return size // (1 if "int8" in path.name else 4)


def count_dir(d: Path) -> tuple[int, list[str]]:
    d = Path(d)
    files = sorted(d.rglob("*.safetensors"))
    if files:
        return sum(safetensors_numel(f) for f in files), [str(f) for f in files]
    onnx = sorted(d.rglob("*.onnx"))
    if onnx:
        fp = [f for f in onnx if "int8" not in f.name] or onnx   # sherpa exports ship fp32 + int8 copies
        return sum(onnx_numel(f) for f in fp), [str(f) for f in fp]
    bins = sorted(d.rglob("*.bin")) + sorted(d.rglob("*.pt"))
    if bins:
        import torch
        tot = 0
        for f in bins:
            sd = torch.load(f, map_location="meta", weights_only=True)
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            tot += sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        return tot, [str(f) for f in bins]
    raise FileNotFoundError(f"no weights found in {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate count-params")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--asr", action="append", default=[])
    ap.add_argument("--extra", action="append", default=[], help="any other model shipped in the image")
    ap.add_argument("--llm-max", type=float, default=5e9)
    ap.add_argument("--total-max", type=float, default=6e9)
    a = ap.parse_args(argv)
    llm, _ = count_dir(Path(a.llm))
    rows = [("LLM", a.llm, llm)]
    for p in a.asr:
        rows.append(("ASR", p, count_dir(Path(p))[0]))
    for p in a.extra:
        rows.append(("extra", p, count_dir(Path(p))[0]))
    total = sum(r[2] for r in rows)
    for kind, p, n in rows:
        print(f"{kind:4} {n / 1e9:7.3f}B  {p}")
    print(f"total {total / 1e9:7.3f}B")
    ok_llm, ok_tot = llm < a.llm_max, total < a.total_max
    print(f"LLM < {a.llm_max / 1e9:g}B: {'ok' if ok_llm else 'OVER'} | total < {a.total_max / 1e9:g}B: "
          f"{'ok' if ok_tot else 'OVER'}")
    raise SystemExit(0 if ok_llm and ok_tot else 1)


if __name__ == "__main__":
    main()
