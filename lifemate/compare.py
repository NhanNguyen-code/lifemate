"""Compare two submissions, e.g. predict.sh output vs the notebook's jupyter_submission.jsonl, plus their `time` field."""
from __future__ import annotations

import argparse

from .data import read_jsonl


def time_stats(rows) -> str:
    t = sorted(r["time"] for r in rows if isinstance(r.get("time"), int))
    if not t:
        return "no time field"
    return (f"mean {sum(t) / len(t):.0f} ms, p50 {t[len(t) // 2]}, p95 {t[int(0.95 * (len(t) - 1))]}, "
            f"max {t[-1]}, over 30 s: {sum(1 for x in t if x > 30000)}")


def compare(a: str, b: str, show: int = 10) -> int:
    ra = {r["id"]: r for r in read_jsonl(a)}
    rb = {r["id"]: r for r in read_jsonl(b)}
    only_a = [i for i in ra if i not in rb]
    only_b = [i for i in rb if i not in ra]
    diff = [i for i in ra if i in rb and (ra[i].get("tool_calls") != rb[i].get("tool_calls")
                                         or ra[i].get("state_hash") != rb[i].get("state_hash"))]
    print(f"{a}: {len(ra)} rows | {b}: {len(rb)} rows")
    print(f"only in first: {len(only_a)} | only in second: {len(only_b)} | different calls/hash: {len(diff)}")
    for name, rows in ((a, ra), (b, rb)):
        print(f"  time {name}: {time_stats(rows.values())}")
    for i in (only_a + only_b + diff)[:show]:
        print(f"  {i}")
    ok = not (only_a or only_b or diff)
    print("same" if ok else "DIFFERENT")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate compare")
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args(argv)
    raise SystemExit(compare(a.first, a.second, a.show))
