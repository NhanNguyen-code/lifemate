"""Grid-search additive biases on the read/write/ask/finish log-probs, scored on val."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate calibrate")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--text", choices=["asr", "oracle"], default="asr")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--exclude-fewshot-ids", default=None)
    ap.add_argument("--fewshot-k", type=int, default=None)
    ap.add_argument("--grid-finish", default="-2,-1,0,1,2")
    ap.add_argument("--grid-ask", default="-1,0,1")
    ap.add_argument("--batch-size", type=int, default=0, help="0: pick from GPU memory")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default="outputs/bias.json")
    a = ap.parse_args(argv)
    from .compat import auto_batch

    from .cli import _ctx, _run_ctx
    from .llm import make_backend
    from .runner import run_split
    from .scorer import Scorer

    a.limit = None
    a.bias = None
    a.temperature = None
    sp, cat, samples = _ctx(a)
    rc = _run_ctx(a, sp, cat)
    backend = make_backend("vllm", model=a.model, lora=a.lora, gpu_memory_utilization=a.gpu_mem)
    scorer = Scorer(sp, cat)

    def evaluate(use_scoring: bool, bf: float = 0.0, ba: float = 0.0) -> dict:
        rc.agent_cfg.use_type_scoring = use_scoring
        rc.agent_cfg.bias_first = {"finish": bf, "ask": ba}
        rc.agent_cfg.bias_later = {"ask": ba}
        res = run_split(rc, backend, samples, batch_size=a.batch_size or auto_batch(), log_every=0)
        rows = [{"id": r["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]} for r in res]
        rep = scorer.score(rows, samples)
        return {"use_type_scoring": use_scoring, "bias_finish_first": bf, "bias_ask": ba,
                "score": rep["score"], "by_category": rep["by_category"]}

    grid = [evaluate(False)]
    print(f"[calib] no bias: {grid[0]['score']}", flush=True)
    for bf in _floats(a.grid_finish):
        for ba in _floats(a.grid_ask):
            r = evaluate(True, bf, ba)
            grid.append(r)
            print(f"[calib] finish_first={bf:+.1f} ask={ba:+.1f}: {r['score']:.2f} {r['by_category']}", flush=True)
    best = max(grid, key=lambda r: r["score"])
    out = {"bias_first": {"finish": best["bias_finish_first"], "ask": best["bias_ask"]},
           "bias_later": {"ask": best["bias_ask"]}, "use_type_scoring": best["use_type_scoring"],
           "best_score": best["score"], "grid": grid}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[calib] best {best['score']:.2f} -> {a.out}"
          + ("" if best["use_type_scoring"] else " (no bias wins, run infer without --bias)"))


if __name__ == "__main__":
    main()
