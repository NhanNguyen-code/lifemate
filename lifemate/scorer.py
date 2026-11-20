"""Local re-implementation of the LifeMate metric.

score = 0 if state_hash != gold, else F1 over calls (bipartite matching; x-match=fuzzy args by token F1 > 0.6).
Gold hashes come from replaying gold calls in a fresh ToolEnv (train only).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .data import Sample, SplitPaths, gold_category, make_env

_W = re.compile(r"\w+", re.UNICODE)


def tokens(s: Any) -> list[str]:
    # as BTC scores fuzzy args: lowercase + Unicode NFC
    return _W.findall(unicodedata.normalize("NFC", str(s)).lower())


def token_f1(a: Any, b: Any) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def args_match(pa: dict, ga: dict, fuzzy: set[str], thr: float = 0.6, strict_keys: bool = True) -> bool:
    pa, ga = pa or {}, ga or {}
    if strict_keys and set(pa) != set(ga):
        return False
    for k, gv in ga.items():
        if k not in pa:
            return False
        if k in fuzzy:
            if token_f1(pa[k], gv) <= thr:
                return False
        elif pa[k] != gv:
            return False
    return True


def mismatched_params(pa: dict, ga: dict, fuzzy: set[str], thr: float = 0.6) -> set[str]:
    out = set()
    for k in set(pa) | set(ga):
        if k not in pa or k not in ga:
            out.add(k)
        elif k in fuzzy:
            if token_f1(pa[k], ga[k]) <= thr:
                out.add(k)
        elif pa[k] != ga[k]:
            out.add(k)
    return out


def call_match(p: dict, g: dict, fuzzy_map: dict[str, set[str]], **kw) -> bool:
    return p.get("name") == g.get("name") and args_match(
        p.get("arguments") or {}, g.get("arguments") or {}, fuzzy_map.get(g.get("name"), set()), **kw)


def max_matching(pred: list[dict], gold: list[dict], fuzzy_map, **kw) -> int:
    adj = [[j for j, g in enumerate(gold) if call_match(p, g, fuzzy_map, **kw)] for p in pred]
    match_g = [-1] * len(gold)

    def try_(i, seen):                  # augmenting path (Kuhn)
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_g[j] < 0 or try_(match_g[j], seen):
                match_g[j] = i
                return True
        return False

    return sum(1 for i in range(len(pred)) if try_(i, set()))


def f1_score(tp: int, n_pred: int, n_gold: int) -> float:
    if n_pred == 0 and n_gold == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_gold
    return 2 * p * r / (p + r)


def diagnose(pred: list[dict], gold: list[dict], cat: Catalog, hash_ok: bool, fuzzy_map) -> list[str]:
    meta = cat.meta_tools
    if not gold:
        return ["oos_false_action"] if pred else []
    if not pred:
        return ["no_action"]
    tags: list[str] = []
    kind = lambda n: "ask" if n in meta else (cat.kind(n) or "write")
    gp, gg = defaultdict(list), defaultdict(list)
    for c in pred:
        gp[kind(c["name"])].append(c)
    for c in gold:
        gg[kind(c["name"])].append(c)
    for k in ("write", "read", "ask"):
        cp = Counter(c["name"] for c in gp[k])
        cg = Counter(c["name"] for c in gg[k])
        for n, v in (cg - cp).items():
            tags.append(f"missing_{k}:{n}")
        for n, v in (cp - cg).items():
            tags.append(f"extra_{k}:{n}")
        for n in sorted(set(cp) & set(cg)):
            ps = [c for c in gp[k] if c["name"] == n]
            gs = [c for c in gg[k] if c["name"] == n]
            if max_matching(ps, gs, fuzzy_map) < min(len(ps), len(gs)):
                fz = fuzzy_map.get(n, set())
                bad: set[str] = set()
                for g in gs:
                    diffs = [mismatched_params(p.get("arguments") or {}, g.get("arguments") or {}, fz) for p in ps]
                    best = min(diffs, key=len)
                    bad |= best
                for a in sorted(bad):
                    tags.append(f"{k}_arg:{n}.{a}")
    if not hash_ok and not any(t.startswith(("missing_write", "extra_write", "write_arg")) for t in tags):
        tags.append("hash_mismatch_other")
    return tags


class Scorer:
    def __init__(self, sp: SplitPaths, catalog: Catalog, cache: str | Path | None = None,
                 strict_keys: bool = True, fuzzy_thr: float = 0.6):
        self.sp = sp
        self.cat = catalog
        self.fuzzy = catalog.fuzzy_map()
        self.kw = dict(strict_keys=strict_keys, thr=fuzzy_thr)
        self.cache_path = Path(cache) if cache else None
        self._hash: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self._hash = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def gold_hash(self, s: Sample) -> str:
        if s.id not in self._hash:
            env = make_env(self.sp, s.id)
            for c in s.gold_calls or []:
                try:
                    env.call(c["name"], dict(c.get("arguments") or {}))
                except Exception as ex:
                    print(f"[scorer] {s.id}: gold call {c['name']} raised {ex}")
            self._hash[s.id] = env.get_state_hash()
        return self._hash[s.id]

    def save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._hash), encoding="utf-8")

    def score_one(self, s: Sample, pred_calls: list[dict], pred_hash: str) -> dict:
        gold = s.gold_calls or []
        hash_ok = pred_hash == self.gold_hash(s)
        tp = max_matching(pred_calls, gold, self.fuzzy, **self.kw)
        f1 = f1_score(tp, len(pred_calls), len(gold))
        score = f1 if hash_ok else 0.0
        return {"id": s.id, "score": score, "hash_ok": hash_ok, "f1": f1, "tp": tp,
                "n_pred": len(pred_calls), "n_gold": len(gold),
                "category": gold_category(s, self.cat.meta_tools),
                "tags": [] if score == 1.0 else diagnose(pred_calls, gold, self.cat, hash_ok, self.fuzzy)}

    def score(self, rows: list[dict], samples: list[Sample], only_submitted: bool = False) -> dict:
        by_id = {r["id"]: r for r in rows}
        per = []
        for s in samples:
            r = by_id.get(s.id)
            if r is None:
                if only_submitted:
                    continue
                r = {"tool_calls": [], "state_hash": ""}
            per.append(self.score_one(s, r.get("tool_calls") or [], r.get("state_hash") or ""))
        self.save_cache()
        return summarize(per)


def summarize(per: list[dict]) -> dict:
    n = len(per) or 1
    cats = defaultdict(list)
    for p in per:
        cats[p["category"]].append(p["score"])
    tags = Counter(t.split(":")[0] for p in per for t in p["tags"])
    detail = Counter(t for p in per for t in p["tags"])
    return {
        "n": len(per),
        "score": round(100 * sum(p["score"] for p in per) / n, 2),
        "hash_ok": round(100 * sum(p["hash_ok"] for p in per) / n, 2),
        "by_category": {k: {"n": len(v), "score": round(100 * sum(v) / max(1, len(v)), 2)}
                        for k, v in sorted(cats.items())},
        "error_types": dict(tags.most_common()),
        "top_errors": dict(detail.most_common(30)),
        "per_sample": per,
    }


def format_report(rep: dict) -> str:
    lines = [f"score {rep['score']:.2f} on {rep['n']} samples | hash match {rep['hash_ok']:.1f}%"]
    for k, v in rep["by_category"].items():
        lines.append(f"  {k:>7}: {v['score']:6.2f}  (n={v['n']})")
    if rep["error_types"]:
        lines.append("error types: " + ", ".join(f"{k}={v}" for k, v in rep["error_types"].items()))
    if rep["top_errors"]:
        lines.append("top errors:")
        for k, v in list(rep["top_errors"].items())[:15]:
            lines.append(f"  {v:4d}  {k}")
    return "\n".join(lines)
