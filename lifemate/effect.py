"""What a run's options did to the score, counted only where they acted (two scored runs of the same samples).

Two runs of one model differ on about one sample in eleven even where nothing was changed: vLLM answers a prompt
slightly differently depending on what shares its batch, and a near tie flips. The difference of two totals is
therefore mostly noise (about +-1 point on 600 samples). The trace says on which samples an option did something;
the score change there is its effect, the change on all other samples shows how large the noise is.

    python -m lifemate effect --base base.report.json --run new.report.json --trace new.trace.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NESTED_MARK = "nằm trong"          # guard: an ask for a field of a list item, sent back once


def touched(row: dict) -> list[str]:
    """What acted on this sample, read from its trace events."""
    out = []
    for e in row.get("events") or []:
        kind, v = e.get("kind"), e.get("vote") or {}
        if e.get("auto"):
            out.append("search by the agent")
        if kind in ("hold", "retry_none") or e.get("forced"):
            out.append("search rewritten")
        if v and v.get("greedy") is False:
            out.append("vote changed the action")
        if v.get("over"):
            out.append("vote kept the greedy action")
        if e.get("ask_fixed") or (kind == "reject" and NESTED_MARK in str(e.get("error"))):
            out.append("ask parameter")
        if e.get("consult"):                   # the second adapter's action ran instead of a stop
            out.append(f"consult taken at {e['consult'].get('on')}")
        if kind in ("lookup_hold", "lookup_none") or e.get("lookup"):
            out.append("call the guard named, forced")
        if e.get("swap"):
            out.append("ID instead of a raw value")
    return list(dict.fromkeys(out))


def effect(base: dict, run: dict, trace: list[dict]) -> dict:
    pa = {p["id"]: p for p in base["per_sample"]}
    pb = {p["id"]: p for p in run["per_sample"]}
    tr = {r["id"]: r for r in trace}
    ids = [i for i in pa if i in pb]
    n = max(1, len(ids))
    groups: dict[str, list[str]] = {}
    rest = []
    for i in ids:
        labels = touched(tr.get(i) or {})
        for lab in labels:
            groups.setdefault(lab, []).append(i)
        if not labels:
            rest.append(i)

    def line(members: list[str]) -> dict:
        d = [pb[i]["score"] - pa[i]["score"] for i in members]
        return {"samples": len(members), "points": round(100 * sum(d) / n, 2),
                "up": sum(x > 1e-9 for x in d), "down": sum(x < -1e-9 for x in d),
                "hash": sum(bool(pb[i]["hash_ok"]) for i in members) - sum(bool(pa[i]["hash_ok"]) for i in members)}

    acted = sorted({i for m in groups.values() for i in m})
    return {"n": len(ids), "total": round(run["score"] - base["score"], 2), "acted": line(acted),
            "not_touched": line(rest), "by_option": {k: line(v) for k, v in groups.items()}}


def acted_on(base: dict, run: dict, trace: list[dict], labels) -> dict:
    """The score change on the samples where any of `labels` (see touched) acted, each sample counted once;
    points are over all samples both runs scored."""
    pa = {p["id"]: p for p in base["per_sample"]}
    pb = {p["id"]: p for p in run["per_sample"]}
    n = max(1, sum(1 for i in pa if i in pb))
    ids = list(dict.fromkeys(r["id"] for r in trace if r.get("id") in pa and r.get("id") in pb
                             and any(lab in touched(r) for lab in labels)))
    d = [pb[i]["score"] - pa[i]["score"] for i in ids]
    return {"samples": len(ids), "points": round(100 * sum(d) / n, 2),
            "up": sum(x > 1e-9 for x in d), "down": sum(x < -1e-9 for x in d)}


def format_effect(r: dict, name: str = "") -> str:
    def row(label: str, x: dict) -> str:
        return (f"  {label:30s} {x['samples']:4d} samples  {x['points']:+6.2f} points  "
                f"(better {x['up']}, worse {x['down']}, hash {x['hash']:+d})")

    lines = [f"{name or 'run'}: total {r['total']:+.2f} on {r['n']} samples",
             row("where an option acted", r["acted"]),
             row("everywhere else (noise)", r["not_touched"])]
    lines += [row("· " + k, v) for k, v in sorted(r["by_option"].items())]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate effect")
    ap.add_argument("--base", required=True, help="report.json of the run without the options")
    ap.add_argument("--run", required=True, nargs="+", help="report.json of each run with options")
    ap.add_argument("--trace", nargs="*", default=None, help="their traces (default: next to each report)")
    a = ap.parse_args(argv)
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    traces = a.trace or [r.replace(".report.json", ".trace.jsonl") for r in a.run]
    for rep, tr in zip(a.run, traces):
        rows = [json.loads(x) for x in Path(tr).read_text(encoding="utf-8").splitlines() if x.strip()]
        res = effect(base, json.loads(Path(rep).read_text(encoding="utf-8")), rows)
        print(format_effect(res, Path(rep).name.replace(".report.json", "")), flush=True)


if __name__ == "__main__":
    main()
