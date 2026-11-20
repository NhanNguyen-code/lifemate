"""Tool catalog: parse the JSON schemas and render them compactly for the prompt."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIND_VI = {"read": "đọc", "write": "ghi", "meta": "hệ thống"}


@dataclass
class ParamSpec:
    name: str
    raw: dict
    required: bool = False

    @property
    def type(self) -> str | None:
        t = self.raw.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), t[0] if t else None)
        return t

    @property
    def enum(self) -> list | None:
        return self.raw.get("enum")

    @property
    def ref(self) -> str | None:
        return self.raw.get("x-ref")

    @property
    def fuzzy(self) -> bool:
        return str(self.raw.get("x-match", "")).lower() == "fuzzy"

    @property
    def description(self) -> str:
        return str(self.raw.get("description") or "")

    @property
    def fmt(self) -> str | None:
        return self.raw.get("format")


@dataclass
class ToolSpec:
    name: str
    app: str
    kind: str
    description: str
    params: dict[str, ParamSpec]
    required: list[str]
    one_of: list[list[str]]
    returns: str
    raw: dict

    @property
    def short(self) -> str:
        return self.name.split(".", 1)[1] if "." in self.name else self.name

    def fuzzy_keys(self) -> set[str]:
        return {k for k, p in self.params.items() if p.fuzzy}

    def ref_keys(self) -> dict[str, str]:
        return {k: p.ref for k, p in self.params.items() if p.ref}


@dataclass
class AppSpec:
    id: str
    name_vi: str
    group: str
    description: str
    tools: list[str] = field(default_factory=list)


def _norm_one_of(v) -> list[list[str]]:
    if not v:
        return []
    out = []
    for g in v:
        if isinstance(g, str):
            out.append([g])
        elif isinstance(g, (list, tuple)):
            out.append([str(x) for x in g])
        elif isinstance(g, dict) and "required" in g:
            out.append([str(x) for x in g["required"]])
    return out


class Catalog:
    def __init__(self, data: dict):
        self.data = data
        self.apps: dict[str, AppSpec] = {}
        self.tools: dict[str, ToolSpec] = {}
        for a in data.get("apps", []):
            aid = a.get("id") or a.get("name")
            app = AppSpec(aid, a.get("name_vi") or aid, a.get("group") or "", a.get("description") or "")
            for t in a.get("tools", []):
                name = t["name"] if "." in t["name"] else f"{aid}.{t['name']}"
                params = t.get("parameters") or {}
                props = params.get("properties") or {}
                req = list(params.get("required") or [])
                ps = {k: ParamSpec(k, v or {}, k in req) for k, v in props.items()}
                spec = ToolSpec(
                    name=name, app=t.get("app") or aid, kind=str(t.get("kind") or "write").lower(),
                    description=str(t.get("description") or ""), params=ps, required=req,
                    one_of=_norm_one_of(params.get("x-one-of") or params.get("x_one_of")),
                    returns=t.get("returns") if isinstance(t.get("returns"), str)
                    else json.dumps(t.get("returns"), ensure_ascii=False) if t.get("returns") else "",
                    raw=t)
                self.tools[name] = spec
                app.tools.append(name)
            self.apps[aid] = app

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy_data(self) -> dict:
        return copy.deepcopy(self.data)

    @property
    def meta_tools(self) -> set[str]:
        return {n for n, t in self.tools.items() if t.kind == "meta"}

    def kind(self, name: str) -> str | None:
        t = self.tools.get(name)
        return t.kind if t else None

    def fuzzy_map(self) -> dict[str, set[str]]:
        return {n: t.fuzzy_keys() for n, t in self.tools.items()}

    def allowed_tools(self, installed: list[str]) -> list[str]:
        seen, out = set(), []
        for n in list(installed) + sorted(self.meta_tools):
            if n in self.tools and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def ask_format(self) -> "AskFormat | None":
        for n in sorted(self.meta_tools):
            if "ask" in n:
                return AskFormat.from_spec(self.tools[n])
        return None

    def render_tools(self, names: list[str], app_order: list[str] | None = None) -> str:
        by_app: dict[str, list[str]] = {}
        for n in names:
            t = self.tools.get(n)
            if t:
                by_app.setdefault(t.app, []).append(n)
        order = [a for a in (app_order or []) if a in by_app] + [a for a in by_app if a not in (app_order or [])]
        lines = []
        for aid in order:
            app = self.apps.get(aid)
            head = f"## {aid}"
            if app:
                extra = " — ".join(x for x in [app.name_vi if app.name_vi != aid else "", app.description] if x)
                if extra:
                    head += f" ({extra})"
            lines.append(head)
            for n in by_app[aid]:
                lines.append(render_tool(self.tools[n]))
        return "\n".join(lines)


def _type_str(p: ParamSpec) -> str:
    t = p.type or "any"
    if t == "array":
        it = (p.raw.get("items") or {}).get("type")
        t = f"array[{it}]" if it else "array"
    return t


def render_param(p: ParamSpec) -> str:
    bits = [f"{p.name}{'*' if p.required else ''}: {_type_str(p)}"]
    if p.enum:
        bits.append("∈{" + "|".join(json.dumps(e, ensure_ascii=False) for e in p.enum) + "}")
    rng = []
    if "minimum" in p.raw:
        rng.append(f">={p.raw['minimum']}")
    if "maximum" in p.raw:
        rng.append(f"<={p.raw['maximum']}")
    if rng:
        bits.append(" ".join(rng))
    if p.fmt:
        bits.append(f"format={p.fmt}")
    if p.ref:
        bits.append(f"ID lấy từ {p.ref}")
    if p.fuzzy:
        bits.append("văn bản tự do")
    s = " ".join(bits)
    if p.description:
        s += f" — {p.description}"
    return s


def render_tool(t: ToolSpec) -> str:
    out = [f"- {t.name} [{KIND_VI.get(t.kind, t.kind)}]: {t.description}".rstrip()]
    for p in t.params.values():
        out.append(f"    · {render_param(p)}")
    if t.one_of:
        out.append("    · bắt buộc một trong: " + " | ".join("+".join(g) for g in t.one_of))
    if t.returns:
        r = t.returns if len(t.returns) <= 160 else t.returns[:157] + "..."
        out.append(f"    → trả về: {r}")
    return "\n".join(out)


@dataclass
class AskFormat:
    tool_name: str
    tool_param: str | None
    missing_param: str | None
    missing_is_list: bool

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "AskFormat":
        # official format is {"tool_name": str, "params": [str]}; read it off the schema in case it changes
        tool_param, missing_param, is_list = None, None, True
        for k, p in spec.params.items():
            lk = k.lower()
            if p.type == "array" and missing_param is None:
                missing_param, is_list = k, True
            elif p.type == "string" and ("tool" in lk or "intent" in lk or "function" in lk) and tool_param is None:
                tool_param = k
        if missing_param is None:
            for k, p in spec.params.items():
                if k != tool_param and ("missing" in k.lower() or "param" in k.lower() or "field" in k.lower()):
                    missing_param, is_list = k, p.type == "array"
                    break
        return cls(spec.name, tool_param, missing_param, is_list)

    def build(self, tool: str, missing: list[str]) -> dict:
        args: dict[str, Any] = {}
        if self.tool_param:
            args[self.tool_param] = tool
        if self.missing_param:
            args[self.missing_param] = list(missing) if self.missing_is_list else ", ".join(missing)
        return args

    def parse(self, args: dict) -> tuple[str | None, list[str]]:
        tool = args.get(self.tool_param) if self.tool_param else None
        miss = args.get(self.missing_param) if self.missing_param else None
        if isinstance(miss, str):
            miss = [m.strip() for m in miss.split(",") if m.strip()]
        return tool, list(miss or [])
