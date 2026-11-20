"""Per-sample score changes between two scored runs of the same samples (`score --report` files)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate diff-runs")
    ap.add_argument("--a", required=True, help="report.json of the reference run")
    ap.add_argument("--b", required=True, help="report.json of the new run")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args(argv)
    ra, rb = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b))
    pa = {p["id"]: p for p in ra["per_sample"]}
    pb = {p["id"]: p for p in rb["per_sample"]}
    ids = [i for i in pa if i in pb]
    d = [(pb[i]["score"] - pa[i]["score"], i) for i in ids]
    better = sorted((x for x in d if x[0] > 1e-9), reverse=True)
    worse = sorted(x for x in d if x[0] < -1e-9)
    print(f"{len(ids)} samples | score {ra['score']:.2f} -> {rb['score']:.2f} | better {len(better)}, worse {len(worse)}")
    for title, rows, rep in (("worse", worse, pb), ("better", better, pa)):
        print(f"-- {title} (tags of the {'new' if rep is pb else 'old'} run)")
        for _, i in rows[: a.show]:
            print(f"  {i}  {pa[i]['score']:.2f} -> {pb[i]['score']:.2f}  {rep[i]['tags']}")
    # error tags that appear only in the new run's worse samples / vanish in its better ones
    came = Counter(t for _, i in worse for t in set(pb[i]["tags"]) - set(pa[i]["tags"]))
    gone = Counter(t for _, i in better for t in set(pa[i]["tags"]) - set(pb[i]["tags"]))
    for title, c in (("tags that came with the worse samples", came), ("tags gone in the better samples", gone)):
        print(f"-- {title}: " + (", ".join(f"{t} {n}" for t, n in c.most_common(a.show)) or "none"))


if __name__ == "__main__":
    main()
