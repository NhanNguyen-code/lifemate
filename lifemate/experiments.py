"""Day-1 checks on train: dataset stats, format probe, and four experiments on how the env/scorer behave."""
from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from pathlib import Path

from .catalog import Catalog
from .data import Sample, SplitPaths, gold_category, is_user_turn, make_env
from .scorer import Scorer
from .textnorm import strip_diacritics
from .timeutil import FormatBook, TIME_SHAPES


def safe_call(env, name: str, args: dict):
    try:
        return env.call(name, args)
    except Exception as ex:
        return {"ok": False, "error": f"exception: {ex}"}


def _replay(sp: SplitPaths, sid: str, calls: list[dict]):
    env = make_env(sp, sid)
    res = [safe_call(env, c["name"], copy.deepcopy(c.get("arguments") or {})) for c in calls]
    return env.get_state_hash(), res


def stats(sp: SplitPaths, cat: Catalog, samples: list[Sample], others: dict[str, Catalog] | None = None) -> dict:
    meta = cat.meta_tools
    cats = Counter(gold_category(s, meta) for s in samples)
    n_calls = [len(s.gold_calls or []) for s in samples if s.gold_calls is not None]
    kinds = Counter((cat.kind(c["name"]) or "?") for s in samples for c in (s.gold_calls or []))
    app_use = Counter(c["name"].split(".")[0] for s in samples for c in (s.gold_calls or []) if c["name"] not in meta)
    hist = [len(s.user_turns()) for s in samples]
    after_ask = sum(1 for s in samples if s.history and any(
        c["name"] in meta for t in s.history[-1:] for c in (t.get("tool_calls") or [])))
    fb = FormatBook()
    for s in samples:
        fb.observe(s.gold_calls or [])
    time_params = {f"{t}.{p}": dict(c.most_common(3)) for (t, p), c in fb.by_param.items()
                   if any(k in TIME_SHAPES for k in c)}
    ask_examples = [c for s in samples for c in (s.gold_calls or []) if c["name"] in meta][:5]
    first_read = Counter()
    for s in samples:
        g = s.gold_calls or []
        if g:
            first_read[cat.kind(g[0]["name"]) or "?"] += 1
    rep = {
        "split": sp.split, "n": len(samples), "categories": dict(cats),
        "calls_per_sample": {"mean": round(statistics.mean(n_calls), 2) if n_calls else 0,
                             "max": max(n_calls) if n_calls else 0,
                             "hist": dict(sorted(Counter(n_calls).items()))},
        "call_kinds": dict(kinds), "first_call_kind": dict(first_read),
        "n_apps_catalog": len([a for a in cat.apps if a != "sys"]),
        "app_usage": dict(app_use.most_common()),
        "installed_apps_per_sample": round(statistics.mean(len(s.installed_apps) for s in samples), 2) if samples else 0,
        "user_turns_in_history": dict(sorted(Counter(hist).items())),
        "history_ends_with_ask": after_ask,
        "fuzzy_params": sorted(f"{t}.{k}" for t, ks in cat.fuzzy_map().items() for k in ks),
        "xref_params": sorted(f"{t.name}.{k} -> {r}" for t in cat.tools.values() for k, r in t.ref_keys().items()),
        "one_of_tools": sorted(t.name for t in cat.tools.values() if t.one_of),
        "ask_format": cat.ask_format().__dict__ if cat.ask_format() else None,
        "ask_examples": ask_examples,
        "time_param_shapes": time_params,
    }
    if others:
        mine = set(cat.apps)
        rep["other_splits"] = {name: {"apps_not_in_this_split": sorted(set(oc.apps) - mine),
                                      "n_apps": len(oc.apps)} for name, oc in others.items()}
    return rep


def data_format(samples: list[Sample]) -> dict:
    # field names and types only, no content
    top, ctx, label = Counter(), Counter(), Counter()
    turn_keys = {"user": Counter(), "assistant": Counter()}
    roles, at_type, tool_entry = Counter(), Counter(), Counter()
    final_ok = hist_total = hist_ok = 0
    for s in samples:
        top.update(s.raw.keys())
        ctx.update(s.context.keys())
        label.update((s.label or {}).keys())
        at_type[type(s.raw.get("audio_text")).__name__] += 1
        for t in s.history:
            roles[str(t.get("role"))] += 1
            turn_keys["user" if is_user_turn(t) else "assistant"].update(t.keys())
        for spec in s.installed_apps.values():
            tools = spec.get("tools", []) if isinstance(spec, dict) else (spec or [])
            for t in tools[:1]:
                tool_entry[type(t).__name__] += 1
        f, h = s.transcripts()
        final_ok += f is not None
        hist_total += len(h)
        hist_ok += sum(x is not None for x in h)
    return {"sample_keys": dict(top), "context_keys": dict(ctx), "label_keys": dict(label),
            "audio_text_type": dict(at_type), "history_roles": dict(roles),
            "history_turn_keys": {k: dict(v) for k, v in turn_keys.items()},
            "installed_tool_entry_type": dict(tool_entry),
            "transcript_final_found": f"{final_ok}/{len(samples)}",
            "transcript_history_found": f"{hist_ok}/{hist_total}"}


def ask_param_order(cat: Catalog, samples: list[Sample]) -> dict:
    ask = cat.ask_format()
    if ask is None:
        return {}
    n = schema = required = 0
    target_installed = n_ask = 0
    for s in samples:
        for c in s.gold_calls or []:
            if c["name"] != ask.tool_name:
                continue
            tool, miss = ask.parse(c.get("arguments") or {})
            spec = cat.tools.get(tool or "")
            n_ask += 1
            target_installed += (tool or "") in set(s.installed_tool_names())
            if spec is None or len(miss) < 2 or any(m not in spec.params for m in miss):
                continue
            n += 1
            props = list(spec.params)
            schema += miss == sorted(miss, key=props.index)
            req = spec.required
            required += miss == sorted(miss, key=lambda m: (req.index(m) if m in req else len(req), props.index(m)))
    return {"asks": n_ask, "target_tool_installed": f"{target_installed}/{n_ask}",
            "multi_param_asks": n,
            "schema_order": round(schema / n, 3) if n else None,
            "required_order": round(required / n, 3) if n else None,
            "note": "if schema_order is 1.0 set guard.ask_param_order = \"schema\" in configs/default.json"}


def exp_hash_fuzzy(sp: SplitPaths, cat: Catalog, samples: list[Sample], limit: int = 60) -> dict:
    fz = cat.fuzzy_map()
    tested, changed_append, changed_case = 0, 0, 0
    examples = []
    for s in samples:
        gold = s.gold_calls or []
        idx = next(((i, k) for i, c in enumerate(gold) if cat.kind(c["name"]) == "write"
                    for k in fz.get(c["name"], ()) if isinstance((c.get("arguments") or {}).get(k), str)), None)
        if idx is None:
            continue
        i, k = idx
        h0, _ = _replay(sp, s.id, gold)
        a = copy.deepcopy(gold)
        a[i]["arguments"][k] = a[i]["arguments"][k] + " nhé"
        h1, _ = _replay(sp, s.id, a)
        b = copy.deepcopy(gold)
        v = b[i]["arguments"][k]
        b[i]["arguments"][k] = (v[:1].upper() + v[1:] + ".") if v else v
        h2, _ = _replay(sp, s.id, b)
        tested += 1
        changed_append += h1 != h0
        changed_case += h2 != h0
        if len(examples) < 3:
            examples.append({"id": s.id, "tool": gold[i]["name"], "param": k, "value": v})
        if tested >= limit:
            break
    verdict = ("n/a (no fuzzy args in write calls)" if not tested else
               "hash depends on fuzzy text: free-text args must match gold exactly to get the hash"
               if changed_append / tested > 0.5 else
               "hash ignores fuzzy text: approximate free text is enough")
    return {"tested": tested,
            "hash_changed_when_appending_word": round(changed_append / max(1, tested), 3),
            "hash_changed_when_case_or_punct": round(changed_case / max(1, tested), 3),
            "examples": examples, "verdict": verdict}


def _found(res) -> bool:
    if not isinstance(res, dict) or not res.get("ok"):
        return False
    for v in res.values():
        if isinstance(v, list):
            return len(v) > 0
    return True


def exp_search_tolerance(sp: SplitPaths, cat: Catalog, samples: list[Sample], limit: int = 60) -> dict:
    tested = 0
    ok = Counter()
    for s in samples:
        for c in s.gold_calls or []:
            spec = cat.tools.get(c["name"])
            if not spec or spec.kind != "read":
                continue
            args = c.get("arguments") or {}
            keys = [k for k, v in args.items() if isinstance(v, str) and k in spec.params
                    and not spec.params[k].ref and not spec.params[k].enum and any(ch.isalpha() for ch in v)]
            if not keys:
                continue
            k = keys[0]
            v = args[k]
            if strip_diacritics(v) == v and v.lower() == v:
                continue
            base = _replay(sp, s.id, [c])[1][0]
            if not _found(base):
                continue
            tested += 1
            for tag, nv in (("no_tones", strip_diacritics(v)), ("lowercase", v.lower()),
                            ("no_tones_lower", strip_diacritics(v).lower()),
                            ("last_word", v.split()[-1] if len(v.split()) > 1 else v)):
                a = dict(args)
                a[k] = nv
                r = _replay(sp, s.id, [{"name": c["name"], "arguments": a}])[1][0]
                ok[tag] += _found(r)
            if tested >= limit:
                break
        if tested >= limit:
            break
    rates = {k: round(v / max(1, tested), 3) for k, v in ok.items()}
    verdict = ("n/a" if not tested else
               "search is tone-insensitive: ASR tone errors in names are mostly harmless"
               if rates.get("no_tones", 0) > 0.8 else
               "search is tone-sensitive: names need fixing before search")
    return {"tested": tested, "still_found_rate": rates, "verdict": verdict}


def exp_read_penalty(sp: SplitPaths, cat: Catalog, samples: list[Sample], scorer: Scorer, limit: int = 80) -> dict:
    extra, missing, n = [], [], 0
    for s in samples:
        gold = s.gold_calls or []
        reads = [i for i, c in enumerate(gold) if cat.kind(c["name"]) == "read"]
        if not reads:
            continue
        n += 1
        p1 = gold + [gold[reads[0]]]
        h1, _ = _replay(sp, s.id, p1)
        extra.append(scorer.score_one(s, p1, h1)["score"])
        p2 = [c for i, c in enumerate(gold) if i != reads[0]]
        h2, _ = _replay(sp, s.id, p2)
        missing.append(scorer.score_one(s, p2, h2)["score"])
        if n >= limit:
            break
    return {"tested": n,
            "score_with_one_extra_read": round(100 * statistics.mean(extra), 2) if extra else None,
            "score_with_one_missing_read": round(100 * statistics.mean(missing), 2) if missing else None,
            "note": "below 100 means reads count in F1, so the model has to copy gold read habits"}


def exp_sanity(sp: SplitPaths, cat: Catalog, samples: list[Sample], scorer: Scorer) -> dict:
    from .llm import OracleBackend
    from .runner import RunContext, run_split
    rc = RunContext(sp, cat, text_source="oracle")
    res = run_split(rc, OracleBackend({s.id: s for s in samples}, cat), samples, batch_size=32, log_every=0)
    rows = [{"id": r["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]} for r in res]
    oracle = scorer.score(rows, samples)
    rejected = [r["id"] for r in res if any(ev["kind"] == "reject" for ev in r["events"])]
    empty = []
    for s in samples:
        env = make_env(sp, s.id)
        empty.append({"id": s.id, "tool_calls": [], "state_hash": env.get_state_hash()})
    e = scorer.score(empty, samples)
    oos = sum(1 for s in samples if gold_category(s, cat.meta_tools) == "oos")
    return {"oracle_through_agent": oracle["score"], "oracle_guard_rejections": rejected[:20],
            "oracle_failed": [p["id"] for p in oracle["per_sample"] if p["score"] < 1][:20],
            "empty_submission": e["score"], "oos_share": round(100 * oos / max(1, len(samples)), 2)}


def run_all(sp: SplitPaths, cat: Catalog, samples: list[Sample], others: dict[str, Catalog],
            out: str | Path | None = None, limit: int = 60) -> dict:
    scorer = Scorer(sp, cat)
    rep = {"stats": stats(sp, cat, samples, others),
           "format": data_format(samples),
           "ask_param_order": ask_param_order(cat, samples),
           "exp1_hash_fuzzy": exp_hash_fuzzy(sp, cat, samples, limit),
           "exp2_search_tolerance": exp_search_tolerance(sp, cat, samples, limit),
           "exp3_read_penalty": exp_read_penalty(sp, cat, samples, scorer, limit),
           "exp4_sanity": exp_sanity(sp, cat, samples, scorer)}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def print_report(rep: dict) -> str:
    st = rep["stats"]
    L = [f"== stats {st['split']} ({st['n']} samples) ==",
         f"categories: {st['categories']}",
         f"calls/sample: mean {st['calls_per_sample']['mean']}, max {st['calls_per_sample']['max']}, "
         f"hist {st['calls_per_sample']['hist']}",
         f"call kinds: {st['call_kinds']} | first call: {st['first_call_kind']}",
         f"apps in catalog: {st['n_apps_catalog']} | installed apps/sample: {st['installed_apps_per_sample']}",
         f"user turns in history: {st['user_turns_in_history']} | history ending with an ask: "
         f"{st['history_ends_with_ask']}",
         f"fuzzy params: {len(st['fuzzy_params'])} | x-ref: {len(st['xref_params'])} | x-one-of tools: "
         f"{len(st['one_of_tools'])}",
         f"ask format: {st['ask_format']}",
         f"ask examples: {json.dumps(st['ask_examples'][:2], ensure_ascii=False)}",
         f"time param formats: {json.dumps(st['time_param_shapes'], ensure_ascii=False)[:600]}"]
    for name, o in (st.get("other_splits") or {}).items():
        L.append(f"{name} apps not in {st['split']}: {len(o['apps_not_in_this_split'])} "
                 f"{o['apps_not_in_this_split']}")
    L += ["", "== data format (field names/types only) ==", json.dumps(rep.get("format"), ensure_ascii=False),
          "== ask_missing_info param order ==", json.dumps(rep.get("ask_param_order"), ensure_ascii=False)]
    e1, e2, e3, e4 = rep["exp1_hash_fuzzy"], rep["exp2_search_tolerance"], rep["exp3_read_penalty"], rep["exp4_sanity"]
    L += ["", "== exp1: hash vs fuzzy args ==", json.dumps(e1, ensure_ascii=False),
          "== exp2: search with tone/case errors ==", json.dumps(e2, ensure_ascii=False),
          "== exp3: one extra / missing read ==", json.dumps(e3, ensure_ascii=False),
          "== exp4: pipeline sanity ==", json.dumps(e4, ensure_ascii=False)]
    return "\n".join(L)
