"""Counts behind the guard's optional rules, on CPU.

Gold side: what the train labels do, so a rule is only switched on if the labels never break it.
Run side (--trace of a val run): what each rule would have touched in that run. Predicted calls are replayed in a
fresh ToolEnv of the train split to get their results.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .actions import canon_key
from .catalog import Catalog
from .data import load_samples, make_env, oracle_texts, read_jsonl, resolve_split
from .guard import (app_of, canonical_name, collect_scalars, history_calls, history_reads, needed_params, note_call,
                    read_apps_in_history)
from .textnorm import normalize, strip_diacritics
from .timeutil import FormatBook


def _fold(s) -> str:
    return strip_diacritics(str(s)).lower()


def ask_target(cat: Catalog, call: dict):
    tool, params = cat.ask_format().parse(call.get("arguments") or {})
    return cat.tools.get(str(tool or "")), params


def ask_kind(spec, params: list[str]) -> str:
    if spec is None:
        return "unknown tool"
    need = needed_params(spec)
    if not need:
        return "tool without required params"
    hit = [p in need for p in params]
    if hit and all(hit):
        return "required only"
    if any(hit):
        return "required + optional"
    return "optional only"


def blocked_as_optional(spec, params: list[str], asked: set[str] = frozenset()) -> bool:
    need = needed_params(spec) if spec else set()
    return bool(need and params and not any(p in need for p in params)
                and not all(f"{spec.name}.{p}" in asked for p in params))


def blocked_before_read(cat: Catalog, spec, params: list[str], read_apps: set[str], allowed: list[str]) -> bool:
    refs = spec.ref_keys() if spec else {}
    for p in params:
        if p in refs:
            rapp = app_of(refs[p])
            if rapp not in read_apps and any(app_of(t) == rapp and cat.kind(t) == "read" for t in allowed):
                return True
    return False


def is_ask(cat: Catalog, call: dict) -> bool:
    a = cat.ask_format()
    return a is not None and call.get("name") == a.tool_name


def apps_read(cat: Catalog, calls: list[dict]) -> set[str]:
    return {app_of(c["name"]) for c in calls if cat.kind(c["name"]) == "read"}


class Words:
    """Does a value appear in what the user said (as said, or digits after number normalisation)?"""

    def __init__(self, s):
        final, hist = oracle_texts(s)
        raw = " ".join(hist + [final])
        self.folded = _fold(raw)
        self.digits = re.sub(r"\D", "", normalize(raw).text) + "|" + re.sub(r"\D", "", raw)

    def has(self, v: str) -> bool:
        d = re.sub(r"\D", "", v)
        return (len(v) >= 3 and _fold(v) in self.folded) or (len(d) >= 3 and d in self.digits)


def xref_sources(cat: Catalog, s, calls: list[dict], results: list) -> list[tuple[int, str, str]]:
    """(call index, ref, where) for each x-ref value:
    'ref app' | 'same key' (seen under the param's own name elsewhere) | 'other app' | 'user words' | 'nowhere'."""
    by_app: dict[str, set[str]] = {}
    by_key: dict[str, set[str]] = {}
    for c, r in history_calls(s.history):
        note_call(cat, by_app, str(c.get("name") or ""), c.get("arguments"), r, by_key)
    words = Words(s)
    out = []
    for i, c in enumerate(calls):
        spec = cat.tools.get(c["name"])
        for k, ref in (spec.ref_keys() if spec else {}).items():
            v = (c.get("arguments") or {}).get(k)
            for x in (v if isinstance(v, list) else [v]):
                if x is None:
                    continue
                sx = str(x)
                if sx in by_app.get(app_of(ref), ()):
                    where = "ref app"
                elif sx in by_key.get(k, ()):
                    where = "same key"
                elif any(sx in vals for vals in by_app.values()):
                    where = "other app"
                elif words.has(sx):
                    where = "user words"
                else:
                    where = "nowhere"
                out.append((i, ref, where))
        note_call(cat, by_app, c["name"], c.get("arguments"), results[i] if i < len(results) else None, by_key)
    return out


def unused_params(fb: FormatBook, cat: Catalog, call: dict) -> list[str]:
    """Args drop_unused_params would remove from this call."""
    spec = cat.tools.get(call["name"])
    if spec is None or fb.tool_calls.get(call["name"], 0) < 20:
        return []
    keep = needed_params(spec)
    return [k for k in (call.get("arguments") or {}) if k not in keep and fb.param_seen(call["name"], k) == 0]


def history_id_reuse(cat: Catalog, s) -> bool:
    """A gold call passes an ID seen in the history but not in any result of this turn."""
    hist: set[str] = set()
    for c, r in history_calls(s.history):
        collect_scalars(r, hist)
    now: set[str] = set()
    results = s.gold_results
    for i, c in enumerate(s.gold_calls or []):
        spec = cat.tools.get(c["name"])
        for k in (spec.ref_keys() if spec else {}):
            v = (c.get("arguments") or {}).get(k)
            for x in (v if isinstance(v, list) else [v]):
                if x is not None and str(x) in hist and str(x) not in now:
                    return True
        if i < len(results):
            collect_scalars(results[i], now)
    return False


def name_params(fb: FormatBook) -> list[tuple[str, str]]:
    return sorted(k for k in fb.values if fb.names(*k))


def fmt(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "none"


def gold_side(cat: Catalog, samples: list, out: list[str]):
    kinds, n_ask, before_read = Counter(), 0, 0
    n_reads = rep = 0
    rep_samples = set()
    reuse = 0
    where, where_refs = Counter(), Counter()
    fb = FormatBook()
    for s in samples:
        gold = s.gold_calls or []
        fb.observe(gold, cat.ask_format())
        allowed = cat.allowed_tools(s.installed_tool_names())
        hist_apps = read_apps_in_history(cat, s.history)
        cached = history_reads(cat, s.history)
        for i, c in enumerate(gold):
            if is_ask(cat, c):
                spec, params = ask_target(cat, c)
                n_ask += 1
                kinds[ask_kind(spec, params)] += 1
                before_read += blocked_before_read(cat, spec, params, hist_apps | apps_read(cat, gold[:i]), allowed)
            elif cat.kind(c["name"]) == "read":
                n_reads += 1
                if canon_key(c["name"], c.get("arguments") or {}) in cached:
                    rep += 1
                    rep_samples.add(s.id)
        reuse += history_id_reuse(cat, s)
        for i, ref, w in xref_sources(cat, s, gold, s.gold_results):
            where[w] += 1
            if w != "ref app":
                where_refs[f"{gold[i]['name']} <- {ref} ({w})"] += 1
    with_hist = sum(1 for s in samples if s.history)
    out.append(f"== gold labels, train ({len(samples)} samples, {with_hist} with history) ==")
    out.append(f"asks: {n_ask} | " + fmt(kinds))
    out.append(f"  ask_needed_only would block {kinds['optional only']} gold asks")
    out.append(f"  ask_ids_after_read would block {before_read} gold asks")
    out.append(f"reads: {n_reads} | repeat a history read the guard would cache: {rep} (in {len(rep_samples)} samples)")
    out.append(f"samples whose gold reuses an ID from the history: {reuse}")
    out.append(f"x-ref values in gold calls: {fmt(where)}")
    out.append("  (xref_typed blocks 'other app'; the guard already blocks 'user words' and 'nowhere')")
    for k, v in where_refs.most_common(8):
        out.append(f"  {v:4d}  {k}")
    params = name_params(fb)
    out.append("name params (fix_names): " + (", ".join(f"{t}.{p} ({len(fb.names(t, p))} names)" for t, p in params)
                                              or "none"))
    return fb


def run_side(cat: Catalog, sp, by_id: dict, trace: Path, held: set[str], val: set[str], hash_cache: str | None,
             out: list[str]):
    from .scorer import Scorer
    rows = [r for r in read_jsonl(trace) if r["id"] in by_id]
    sc = Scorer(sp, cat, hash_cache)
    fb = FormatBook()                          # what the guard sees in a val run: train minus val
    for s in by_id.values():
        if s.id not in val:
            fb.observe(s.gold_calls or [], cat.ask_format())
    now = after = 0.0
    n_extra = n_ok_ask = n_missed = 0
    extra_held, extra_params = Counter(), Counter()
    rule_opt, rule_read = Counter(), Counter()
    typed = Counter()
    unused = Counter()
    xref_rej, xref_rej_stop = 0, Counter()
    names = Counter()
    cached_reads, cached_in_gold, cached_samples = 0, 0, set()
    reuse_n, reuse_score, reuse_first = 0, 0.0, Counter()
    for r in rows:
        s = by_id[r["id"]]
        gold = s.gold_calls or []
        pred = r.get("tool_calls") or []
        allowed = cat.allowed_tools(s.installed_tool_names())
        base = sc.score_one(s, pred, r.get("state_hash") or "")
        now += base["score"]
        env = make_env(sp, s.id)               # replay to get the results the model saw
        results = []
        for c in pred:
            try:
                results.append(env.call(c["name"], dict(c.get("arguments") or {})))
            except Exception as ex:
                results.append({"ok": False, "error": str(ex)})
        gold_asks = [c for c in gold if is_ask(cat, c)]
        pred_asks = [(i, c) for i, c in enumerate(pred) if is_ask(cat, c)]
        if gold_asks and not pred_asks:
            n_missed += 1
        hist_apps = read_apps_in_history(cat, s.history)
        for i, c in pred_asks:
            spec, params = ask_target(cat, c)
            if not gold_asks:
                kind = "extra"
                n_extra += 1
                extra_held["held-out app" if spec and app_of(spec.name) in held else "app seen in training"] += 1
                for p in params:
                    extra_params[f"{spec.name if spec else '?'}.{p}"] += 1
            elif any(c.get("arguments") == g.get("arguments") for g in gold_asks):
                kind = "matches gold"
                n_ok_ask += 1
            else:
                kind = "wrong ask"
            if blocked_as_optional(spec, params, fb.asked):
                rule_opt[kind] += 1
            if blocked_before_read(cat, spec, params, hist_apps | apps_read(cat, pred[:i]), allowed):
                rule_read[kind] += 1
        for i, ref, w in xref_sources(cat, s, pred, results):
            if w == "other app" and any(app_of(t) == app_of(ref) and cat.kind(t) == "read" for t in allowed):
                typed["call matches gold" if pred[i] in gold else "call not in gold"] += 1
        for c in gold:                         # gold args the rule would drop: certain loss
            unused["gold args dropped"] += len(unused_params(fb, cat, c))
        for c in pred:
            gone = unused_params(fb, cat, c)
            if gone:
                c2 = {**c, "arguments": {k: v for k, v in (c.get("arguments") or {}).items() if k not in gone}}
                unused["pred calls trimmed"] += 1
                unused["... now matching gold"] += c2 in gold and c not in gold
        rej = [e for e in r.get("events") or [] if e.get("kind") == "reject" and "chưa xuất hiện" in str(e.get("error"))]
        if rej:
            xref_rej += 1
            xref_rej_stop[r.get("stop_reason") or "?"] += 1
        words = " | ".join(t for t in oracle_texts(s)[1] + [oracle_texts(s)[0]] if t)
        for c in pred:
            for k, v in (c.get("arguments") or {}).items():
                nm = fb.names(c["name"], k)
                if not nm or not isinstance(v, str):
                    continue
                v2 = canonical_name(v, nm, words)
                gv = {(g.get("arguments") or {}).get(k) for g in gold if g["name"] == c["name"]}
                if v2 == v:
                    names["kept, " + ("right" if v in gv else "wrong")] += 1
                elif v2 in gv:
                    names["fixed"] += 1
                elif v in gv:
                    names["broken"] += 1
                else:
                    names["changed, still wrong"] += 1
        # history-read cache: those reads never reach env, the rest of the run is unchanged
        cache = history_reads(cat, s.history)
        keep = [c for c in pred if not (cat.kind(c["name"]) == "read"
                                        and canon_key(c["name"], c.get("arguments") or {}) in cache)]
        if len(keep) < len(pred):
            cached_samples.add(s.id)
            gone = [c for c in pred if c not in keep]
            cached_reads += len(gone)
            cached_in_gold += sum(1 for c in gone if c in gold)
        after += sc.score_one(s, keep, r.get("state_hash") or "")["score"]
        if history_id_reuse(cat, s):
            reuse_n += 1
            reuse_score += base["score"]
            first = pred[0] if pred else None
            if first is None:
                reuse_first["no call"] += 1
            elif is_ask(cat, first):
                reuse_first["asked first"] += 1
            elif cat.kind(first["name"]) == "read":
                reuse_first["read first"] += 1
            else:
                reuse_first["wrote first"] += 1
    sc.save_cache()
    n = len(rows) or 1
    out.append(f"== run {trace.name} ({len(rows)} samples) ==")
    out.append(f"score {100 * now / n:.2f} | asks: extra {n_extra}, matching gold {n_ok_ask} | gold asks missed {n_missed}")
    out.append(f"extra asks by target app: {fmt(extra_held)}")
    out.append("params of extra asks: " + ", ".join(f"{k} {v}" for k, v in extra_params.most_common(12)))
    out.append(f"ask_needed_only would reject: {fmt(rule_opt)}")
    out.append(f"ask_ids_after_read would reject: {fmt(rule_read)}")
    out.append(f"xref_typed would reject IDs taken from another app: {fmt(typed)}")
    out.append(f"drop_unused_params: {fmt(unused)}")
    out.append(f"samples with an x-ref rejection: {xref_rej}, they ended with: {fmt(xref_rej_stop)}")
    out.append(f"fix_names on predicted names: {fmt(names)}")
    out.append(f"cache_history_reads: {cached_reads} reads in {len(cached_samples)} samples answered from the history "
               f"({cached_in_gold} of them also in gold) -> score {100 * now / n:.2f} -> {100 * after / n:.2f}")
    if reuse_n:
        out.append(f"gold reuses a history ID: {reuse_n} samples, score {100 * reuse_score / reuse_n:.1f}, "
                   f"run started with: {fmt(reuse_first)}")


def public_side(data: str, cat: Catalog, fb: FormatBook, out: list[str]):
    try:
        psp = resolve_split(data, "public_test")
    except FileNotFoundError:
        return
    pcat = Catalog.load(psp.catalog)
    new = sorted(set(pcat.apps) - set(cat.apps))
    ps = load_samples(psp)
    hit = sum(1 for s in ps if set(s.installed_apps) & set(new))
    known = Counter()
    for s in ps:
        for c, _ in history_calls(s.history):
            for k, v in (c.get("arguments") or {}).items():
                nm = fb.names(str(c.get("name")), k)
                if nm and isinstance(v, str):
                    known["known name" if v in nm else "new name"] += 1
    out.append("== public_test ==")
    out.append(f"apps not in train: {len(new)} of {len(pcat.apps)}: {new}")
    out.append(f"samples with one of them installed: {hit} / {len(ps)}")
    out.append(f"names in public history calls vs train names: {fmt(known)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate diag")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default=None, help="make-split output, for held-out apps and val ids")
    ap.add_argument("--trace", default=None, help="trace of a val run (infer --trace)")
    ap.add_argument("--hash-cache", default=None)
    a = ap.parse_args(argv)
    sp = resolve_split(a.data, "train")
    cat = Catalog.load(sp.catalog)
    if cat.ask_format() is None:
        raise SystemExit("no ask tool in the catalog")
    samples = load_samples(sp)
    out: list[str] = []
    fb = gold_side(cat, samples, out)
    if a.trace:
        held, val = set(), set()
        if a.split_file:
            spl = json.loads(Path(a.split_file).read_text(encoding="utf-8"))
            held, val = set(spl.get("held_apps") or []), set(spl.get("val_ids") or [])
        run_side(cat, sp, {s.id: s for s in samples}, Path(a.trace), held, val, a.hash_cache, out)
    public_side(a.data, cat, fb, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
