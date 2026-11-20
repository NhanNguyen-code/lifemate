"""Augmentations for unseen apps: renaming, distractor apps, removing the needed app (-> out of scope)."""
from __future__ import annotations

import copy
import random
import re

from .catalog import AskFormat, Catalog

SYL = ["ka", "lo", "mi", "ra", "tu", "ve", "zo", "ni", "pa", "se", "do", "fi", "gu", "ho", "ju", "be",
       "qa", "wy", "xe", "yo"]


def _rand_id(rng: random.Random, used: set[str], parts: int = 2) -> str:
    while True:
        s = "".join(rng.choice(SYL) for _ in range(parts)) + str(rng.randint(1, 9))
        if s not in used:
            used.add(s)
            return s


def make_rename(cat: Catalog, apps: list[str], rng: random.Random, p_app: float = 1.0,
                p_tool: float = 0.5, p_param: float = 0.0) -> dict:
    used: set[str] = set(cat.apps)
    rn = {"apps": {}, "tools": {}, "params": {}}
    for a in apps:
        if a == "sys" or a not in cat.apps:
            continue
        new_app = _rand_id(rng, used) if rng.random() < p_app else a
        if new_app != a:
            rn["apps"][a] = new_app
        tused: set[str] = set()
        for tname in cat.apps[a].tools:
            short = tname.split(".", 1)[1]
            new_short = ("f_" + _rand_id(rng, tused)) if rng.random() < p_tool else short
            new_full = f"{new_app}.{new_short}"
            if new_full != tname:
                rn["tools"][tname] = new_full
            if p_param > 0:
                pm = {}
                pused: set[str] = set()
                for k in cat.tools[tname].params:
                    if rng.random() < p_param:
                        pm[k] = "p_" + _rand_id(rng, pused)
                if pm:
                    rn["params"][tname] = pm
    return rn


def _ref(v: str, rn: dict) -> str:
    if isinstance(v, str) and "." in v:
        app, rest = v.split(".", 1)
        return f"{rn['apps'].get(app, app)}.{rest}"
    return v


def rename_catalog_data(data: dict, rn: dict) -> dict:
    d = copy.deepcopy(data)
    for app in d.get("apps", []):
        aid = app.get("id") or app.get("name")
        new_aid = rn["apps"].get(aid, aid)
        app["id"] = new_aid
        for t in app.get("tools", []):
            full = t["name"] if "." in t["name"] else f"{aid}.{t['name']}"
            pm = rn["params"].get(full, {})
            t["name"] = rn["tools"].get(full, full if new_aid == aid else f"{new_aid}.{full.split('.', 1)[1]}")
            if "app" in t:
                t["app"] = new_aid
            params = t.get("parameters") or {}
            props = params.get("properties") or {}
            newp = {}
            for k, v in props.items():
                v = dict(v)
                if "x-ref" in v:
                    v["x-ref"] = _ref(v["x-ref"], rn)
                newp[pm.get(k, k)] = v
            if props:
                params["properties"] = newp
            if params.get("required"):
                params["required"] = [pm.get(k, k) for k in params["required"]]
            if params.get("x-one-of"):
                params["x-one-of"] = [[pm.get(k, k) for k in (g if isinstance(g, list) else [g])]
                                      for g in params["x-one-of"]]
    return d


def _tool_name(name: str, rn: dict) -> str:
    if name in rn["tools"]:
        return rn["tools"][name]
    if "." in name:
        app, short = name.split(".", 1)
        if app in rn["apps"]:
            return f"{rn['apps'][app]}.{short}"
    return name


def rename_call(call: dict, rn: dict, ask: AskFormat | None) -> dict:
    name = call["name"]
    pm = rn["params"].get(name, {})
    args = {pm.get(k, k): v for k, v in (call.get("arguments") or {}).items()}
    if ask is not None and name == ask.tool_name:
        tool, missing = ask.parse(args)
        if tool:
            tpm = rn["params"].get(tool, {})
            rest = {k: v for k, v in args.items() if k not in (ask.tool_param, ask.missing_param)}
            args = ask.build(_tool_name(tool, rn), [tpm.get(m, m) for m in missing]) | rest
    return {"name": _tool_name(name, rn), "arguments": args}


def rename_sample_raw(raw: dict, rn: dict, ask: AskFormat | None) -> dict:
    r = copy.deepcopy(raw)
    ctx = r.get("context") or {}
    inst = ctx.get("installed_apps") or {}
    new_inst = {}
    for app, spec in inst.items():
        tools = spec.get("tools", []) if isinstance(spec, dict) else list(spec or [])
        full = [t if "." in t else f"{app}.{t}" for t in tools if isinstance(t, str)]
        new_inst[rn["apps"].get(app, app)] = {"tools": [_tool_name(t, rn) for t in full]}
    ctx["installed_apps"] = new_inst
    for turn in r.get("history") or []:
        if turn.get("tool_calls"):
            turn["tool_calls"] = [rename_call(c, rn, ask) for c in turn["tool_calls"]]
    lab = r.get("label")
    if lab and lab.get("tool_calls") is not None:
        lab["tool_calls"] = [rename_call(c, rn, ask) for c in lab["tool_calls"]]
    return r


def shuffle_installed(raw: dict, rng: random.Random) -> dict:
    r = copy.deepcopy(raw)
    inst = (r.get("context") or {}).get("installed_apps") or {}
    items = list(inst.items())
    rng.shuffle(items)
    new = {}
    for app, spec in items:
        tools = list(spec.get("tools", [])) if isinstance(spec, dict) else list(spec or [])
        rng.shuffle(tools)
        new[app] = {"tools": tools}
    r["context"]["installed_apps"] = new
    return r


def _used_apps(raw: dict, meta: set[str]) -> set[str]:
    return {c["name"].split(".")[0] for c in (raw.get("label") or {}).get("tool_calls") or []
            if c["name"] not in meta}


def add_distractors(raw: dict, cat: Catalog, rng: random.Random, k: int = 1,
                    exclude: set[str] | None = None) -> dict | None:
    inst = raw["context"]["installed_apps"]
    used = _used_apps(raw, cat.meta_tools)
    if not used:
        return None
    groups = {cat.apps[a].group for a in used if a in cat.apps}
    cands = [a for a, spec in cat.apps.items() if a != "sys" and a not in inst and spec.group not in groups
             and a not in (exclude or set())]
    if not cands:
        return None
    r = copy.deepcopy(raw)
    for a in rng.sample(cands, min(k, len(cands))):
        r["context"]["installed_apps"][a] = {"tools": list(cat.apps[a].tools)}
    return r


def remove_app_to_oos(raw: dict, cat: Catalog) -> dict | None:
    lab = raw.get("label") or {}
    gold = lab.get("tool_calls") or []
    if not gold or raw.get("history") or any(c["name"] in cat.meta_tools for c in gold):
        return None
    inst = raw["context"]["installed_apps"]
    write_apps = {c["name"].split(".")[0] for c in gold if cat.kind(c["name"]) == "write"}
    if not write_apps:
        return None
    groups = {cat.apps[a].group for a in write_apps if a in cat.apps}
    # skip if another installed app of the same group could still do it, the empty label would be wrong
    others = [a for a in inst if a not in write_apps and a in cat.apps and cat.apps[a].group in groups]
    if others or not groups:
        return None
    r = copy.deepcopy(raw)
    for a in write_apps:
        r["context"]["installed_apps"].pop(a, None)
    if not r["context"]["installed_apps"]:
        return None
    r["label"] = {"tool_calls": [], "tool_results": [], "final_state": {}}
    return r


_CONNECT = r"(?:\s+(?:là|rằng|nội dung|với nội dung|nói|bảo|ghi|tiêu đề|tên))?"


def drop_slot_to_ask(raw: dict, cat: Catalog, ask: AskFormat | None) -> dict | None:
    if ask is None or not ask.tool_param or raw.get("history"):
        return None
    lab = raw.get("label") or {}
    gold = lab.get("tool_calls") or []
    from .data import Sample
    text = Sample(raw).transcripts()[0] or ""
    for i, c in enumerate(gold):
        spec = cat.tools.get(c["name"])
        if not spec or spec.kind != "write":
            continue
        for k in spec.fuzzy_keys():
            v = (c.get("arguments") or {}).get(k)
            if not (isinstance(v, str) and v and k in spec.required):
                continue
            m = re.search(_CONNECT + r"\s+" + re.escape(v.lower()) + r"\b", text.lower())
            if not m:
                continue
            new_text = (text[:m.start()] + text[m.end():]).strip()
            if len(new_text.split()) < 2:
                continue
            r = copy.deepcopy(raw)
            r["audio_text"] = new_text
            new_calls = gold[:i] + [{"name": ask.tool_name, "arguments": ask.build(c["name"], [k])}]
            new_res = (lab.get("tool_results") or [])[:i] + [{"ok": True}]
            r["label"] = {"tool_calls": new_calls, "tool_results": new_res, "final_state": {}}
            return r
    return None
