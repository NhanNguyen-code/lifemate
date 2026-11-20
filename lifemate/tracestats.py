"""What a run did, read from its trace alone (no labels needed, so it also works on a test split).

The train labels give the yardstick: about 24% of the samples have no write, a read is empty in 0.3% of the calls and
a write never comes back with done=false. A run far from that is giving up or acting on wrong arguments.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

NUDGE_MARK = "chưa được thử lại"


def _kind(catalog, name: str) -> str:
    spec = catalog.tools.get(name) if catalog is not None else None
    return {"meta": "ask"}.get(spec.kind, spec.kind) if spec is not None else "?"


def stats(rows: list[dict], catalog=None) -> dict:
    n = len(rows)
    out: dict = {"n": n, "stop": dict(Counter(r.get("stop_reason") for r in rows))}
    calls = writes = no_write = read_only = no_call = 0
    after_fail: Counter = Counter()
    fail_eps = fail_eps_write = fail_eps_ask = fail_eps_idle = 0
    c = Counter()
    by_tool: Counter = Counter()
    votes = vote_changed = vote_split = 0
    vote_over: Counter = Counter()
    for r in rows:
        tc = r.get("tool_calls") or []
        kinds = [_kind(catalog, x["name"]) for x in tc]
        # a write that came back with done=false changed nothing: it does not count as one
        w = kinds.count("write") - sum(1 for e in (r.get("events") or []) if e.get("kind") == "call" and e.get("noop")
                                       and _kind(catalog, e.get("name")) == "write")
        calls += len(tc)
        writes += w
        no_call += not tc
        no_write += w == 0
        read_only += bool(tc) and w == 0 and "ask" not in kinds
        ev = r.get("events") or []
        failed = False
        for i, e in enumerate(ev):
            v = e.get("vote")
            if v:
                votes += 1
                vote_changed += not v.get("greedy", True)
                vote_split += v.get("top", 0) < v.get("k", 0)
                if v.get("over"):
                    vote_over[v["over"]] += 1
            if e.get("kind") == "call" and e.get("auto"):
                c[f"searches by the agent ({e['auto']})"] += 1
                c["... that found something"] += not e.get("empty")
            if e.get("kind") == "hold":
                c["stops held for a rewritten search"] += 1
            if e.get("kind") == "retry_none":
                c["... no new call written"] += 1
            if e.get("kind") == "call" and e.get("forced"):
                c["... new call run"] += 1
                c["... new call found something"] += not e.get("empty")
            if e.get("kind") == "cached":
                c["repeated calls (cached)"] += 1
            if e.get("kind") == "consult":
                c[f"consult at {e.get('on')}: not taken ({e.get('why') or e.get('type') or 'no action'})"] += 1
            if e.get("consult"):
                c[f"consult at {e['consult'].get('on')}: {_kind(catalog, e.get('name'))} taken"] += 1
            if e.get("by") == "consult" and e.get("kind") == "call":
                c["calls by the consult adapter after a handover"] += 1
            if e.get("kind") == "lookup_hold":
                c[f"calls the guard named, forced (at {e.get('stop')})"] += 1
            if e.get("kind") == "lookup_none":
                c["... no valid call written"] += 1
            if e.get("kind") == "call" and e.get("lookup"):
                c["... forced call run"] += 1
            if e.get("swap"):
                c[f"ID instead of a raw value ({e['swap']})"] += 1
            if e.get("kind") == "ask_deferred":
                c["asks held back"] += 1
                later = [x for x in ev[i + 1:] if x.get("kind") == "call" and not x.get("deferred")]
                c["... followed by other calls"] += bool(later)
            if e.get("kind") == "drop":
                c[f"dropped after an ask ({e.get('why')})"] += 1
            if e.get("kind") == "reject" and NUDGE_MARK in str(e.get("error")):
                c["stops sent back for a retry"] += 1
                nxt = next((x for x in ev[i + 1:]), {})
                c[f"... then {nxt.get('kind', 'end')}"] += 1
            if e.get("kind") == "reject" and str(e.get("error")).startswith("Thiếu thông tin thì chưa dừng"):
                c["finishes sent back (information missing)"] += 1
                nxt = next((x for x in ev[i + 1:]), {})
                c[f"... then {nxt.get('kind', 'end')} "] += 1
            if e.get("cut"):
                c["results cut in the prompt"] += 1
            if e.get("kind") == "call" and (e.get("empty") or e.get("noop")):
                failed = True
                what = "empty read" if e.get("empty") else "no-op call"
                c[what] += 1
                by_tool[e.get("name")] += 1
                nxt = next((x for x in ev[i + 1:]), {})
                nk = nxt.get("kind", "end")
                if nk == "call" and nxt.get("auto"):
                    nk = "search by the agent"
                elif nk == "call":
                    nk = "call, same tool" if nxt.get("name") == e.get("name") else "call, other tool"
                elif nk == "hold":
                    nk = "stop held"
                elif nk == "reject" and NUDGE_MARK in str(nxt.get("error")):
                    nk = "stop sent back"
                after_fail[nk] += 1
        if failed:
            fail_eps += 1
            fail_eps_write += w > 0
            fail_eps_ask += "ask" in kinds and w == 0
            fail_eps_idle += w == 0 and "ask" not in kinds
    out.update({
        "calls_per_sample": round(calls / max(1, n), 2), "writes_per_sample": round(writes / max(1, n), 2),
        "no_write_pct": round(100 * no_write / max(1, n), 1), "read_only_pct": round(100 * read_only / max(1, n), 1),
        "no_call_pct": round(100 * no_call / max(1, n), 1),
        "failed_calls": {"samples": fail_eps, "ended_with_a_write": fail_eps_write, "ended_with_only_an_ask": fail_eps_ask,
                         "ended_without_write_or_ask": fail_eps_idle, "next_event": dict(after_fail.most_common()),
                         "tools": dict(by_tool.most_common(12))},
        "counts": dict(c),
    })
    if votes:
        out["votes"] = {"steps": votes, "not_unanimous": vote_split, "greedy_overruled": vote_changed,
                        "kept_against_the_count": dict(vote_over)}
    return out


def format_stats(s: dict) -> str:
    f = s["failed_calls"]
    lines = [
        f"{s['n']} samples | stop: {s['stop']}",
        f"calls/sample {s['calls_per_sample']} | writes/sample {s['writes_per_sample']} (train labels: 2.29 / 1.12)",
        f"no write {s['no_write_pct']}% (labels 24%) | only reads {s['read_only_pct']}% (labels 8%) | "
        f"no call {s['no_call_pct']}% (labels 7%)",
        f"samples with an empty read or a no-op call: {f['samples']} -> with a write in the end {f['ended_with_a_write']}, "
        f"only an ask {f['ended_with_only_an_ask']}, neither {f['ended_without_write_or_ask']}",
        f"  right after such a call: {f['next_event']}",
        f"  tools: {f['tools']}",
    ]
    for k, v in s["counts"].items():
        lines.append(f"{k}: {v}")
    if "votes" in s:
        v = s["votes"]
        lines.append(f"votes: {v['steps']} steps, not unanimous {v['not_unanimous']}, greedy overruled "
                     f"{v['greedy_overruled']}")
        if v.get("kept_against_the_count"):
            lines.append(f"  kept although the plain count favoured another action: {v['kept_against_the_count']}")
    return "\n".join(lines)


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
