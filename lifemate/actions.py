"""One action per model turn: {"thought"?, "type": read|write|ask, "name", "arguments"} or {"thought"?, "type": "finish"}."""
from __future__ import annotations

import json
from typing import Any

TYPES = ("read", "write", "ask", "finish")
TYPE_PREFIX = '{"type": "'
KIND_TO_TYPE = {"read": "read", "write": "write", "meta": "ask"}


def action_text(a: dict) -> str:
    # an optional short "thought" comes first, so the model reasons before it commits to a tool
    out = {"thought": a["thought"]} if a.get("thought") else {}
    if a.get("type") == "finish":
        out["type"] = "finish"
    else:
        out.update(type=a["type"], name=a["name"], arguments=a.get("arguments") or {})
    return json.dumps(out, ensure_ascii=False)


def finish(thought: str | None = None) -> dict:
    return {"thought": thought, "type": "finish"} if thought else {"type": "finish"}


def call_to_action(call: dict, catalog) -> dict:
    kind = catalog.kind(call["name"]) if catalog is not None else None
    return {"type": KIND_TO_TYPE.get(kind or "", "write"), "name": call["name"],
            "arguments": dict(call.get("arguments") or {})}


def parse_action(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    dec = json.JSONDecoder()
    obj: Any = None
    for extra in ("", "}", "}}", "}}}", '"}}', '"}'):
        try:
            obj, _ = dec.raw_decode(s + extra)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return None
    th = obj.get("thought")
    th = th.strip() if isinstance(th, str) and th.strip() else None
    t = obj.get("type")
    if t == "finish":
        return finish(th)
    name = obj.get("name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    if t not in TYPES:
        t = None
    a = {"type": t, "name": name, "arguments": args}
    return {"thought": th, **a} if th else a


def canon_key(name: str, args: dict) -> str:
    return name + " " + json.dumps(args, ensure_ascii=False, sort_keys=True)
