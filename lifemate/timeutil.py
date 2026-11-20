"""Date table for the prompt, plus learning and forcing the date-time formats seen in train labels."""
from __future__ import annotations

import json

import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
_FIXED_TZ = {"Asia/Ho_Chi_Minh": 7, "Asia/Saigon": 7, "Asia/Bangkok": 7, "Asia/Jakarta": 7,
             "UTC": 0, "Etc/UTC": 0}


def _tz(name: str | None):
    if not name:
        return timezone(timedelta(hours=7))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=_FIXED_TZ.get(name, 7)))


def parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    for cand in (s, s.replace(" ", "T")):
        try:
            return datetime.fromisoformat(cand)
        except ValueError:
            continue
    return None


def now_from_context(ctx: dict) -> datetime:
    dt = parse_dt(str(ctx.get("current_time") or ""))
    tz = _tz(ctx.get("timezone"))
    if dt is None:
        return datetime.now(tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _d(x: date) -> str:
    return f"{WEEKDAYS_VI[x.weekday()]} {x.isoformat()}"


def date_table(ctx: dict) -> str:
    now = now_from_context(ctx)
    today = now.date()
    monday = today - timedelta(days=today.weekday())
    this_week = [monday + timedelta(days=k) for k in range(7)]
    next_week = [monday + timedelta(days=7 + k) for k in range(7)]
    lines = [
        f"Bây giờ: {_d(today)} {now.strftime('%H:%M')} (ISO {now.isoformat()})",
        f"Hôm qua: {_d(today - timedelta(days=1))} | Hôm nay: {_d(today)} | Mai: {_d(today + timedelta(days=1))}"
        f" | Ngày kia/mốt: {_d(today + timedelta(days=2))}",
        "Tuần này: " + ", ".join(_d(x) for x in this_week),
        "Tuần sau: " + ", ".join(_d(x) for x in next_week),
    ]
    nm = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    lines.append(f"Tháng này: {today.month:02d}/{today.year} | Tháng sau: {nm.month:02d}/{nm.year}")
    return "\n".join(lines)


SHAPES = [
    ("datetime_tz", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)$")),
    ("datetime_tz_nosec", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$")),
    ("datetime", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")),
    ("datetime_nosec", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")),
    ("datetime_space", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$")),
    ("date", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("time", re.compile(r"^\d{2}:\d{2}$")),
    ("time_sec", re.compile(r"^\d{2}:\d{2}:\d{2}$")),
]
TIME_SHAPES = {s for s, _ in SHAPES}
DATETIME_SHAPES = {"datetime_tz", "datetime_tz_nosec", "datetime", "datetime_nosec", "datetime_space"}


def shape_of(v: Any) -> str:
    if not isinstance(v, str):
        return type(v).__name__
    for name, rx in SHAPES:
        if rx.match(v):
            return name
    return "text"


def _loose_parse(v: str, ref: datetime):
    v = v.strip()
    dt = parse_dt(v)
    if dt is not None:
        return dt
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{1,2})[:h](\d{2}))?", v)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h, mi = int(m.group(4) or 0), int(m.group(5) or 0)
        return datetime(y, mo, d, h, mi)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2})[:h](\d{1,2})(?::(\d{1,2}))?", v)
    if m:
        y, mo, d, h, mi = (int(m.group(k)) for k in range(1, 6))
        return datetime(y, mo, d, h, mi, int(m.group(6) or 0))
    m = re.fullmatch(r"(\d{1,2})[:h](\d{2})?", v)
    if m:
        return ref.replace(hour=int(m.group(1)), minute=int(m.group(2) or 0), second=0, microsecond=0,
                           tzinfo=None)
    return None


def coerce_shape(v: Any, shape: str, ctx: dict) -> Any:
    if not isinstance(v, str) or shape not in TIME_SHAPES or shape_of(v) == shape:
        return v
    if shape in DATETIME_SHAPES and shape_of(v) == "date":
        return v                        # fromisoformat would make it midnight; the guard decides instead
    now = now_from_context(ctx)
    dt = _loose_parse(v, now)
    if dt is None:
        return v
    if shape in ("datetime_tz", "datetime_tz_nosec"):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=now.tzinfo)
        s = dt.isoformat(timespec="seconds" if shape == "datetime_tz" else "minutes")
        return s
    dt = dt.replace(tzinfo=None)
    if shape == "datetime":
        return dt.isoformat(timespec="seconds")
    if shape == "datetime_nosec":
        return dt.isoformat(timespec="minutes")
    if shape == "datetime_space":
        return dt.strftime("%Y-%m-%d %H:%M")
    if shape == "date":
        return dt.date().isoformat()
    if shape == "time":
        return dt.strftime("%H:%M")
    if shape == "time_sec":
        return dt.strftime("%H:%M:%S")
    return v


TIME_PARAM_RX = re.compile(r"(time|date|_at$|^at$|deadline|start|end|when|due|datetime|thoi_gian|ngay)", re.I)


class FormatBook:

    def __init__(self):
        self.by_param: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.global_time: Counter = Counter()
        self.values: dict[tuple[str, str], Counter] = defaultdict(Counter)    # short strings, for names()
        self.name_sets: dict[tuple[str, str], set[str]] = {}                  # names() as loaded from json
        self.tool_calls: Counter = Counter()
        self.repeatable: set[str] = set()       # tools gold calls twice with the same args (redeem 2 items)
        self.asked: set[str] = set()            # "tool.param" that gold asks the user for

    def observe(self, calls: list[dict], ask=None):
        seen = set()
        for c in calls:
            args = c.get("arguments") or {}
            self.tool_calls[c["name"]] += 1
            key = c["name"] + json.dumps(args, ensure_ascii=False, sort_keys=True)
            if key in seen:
                self.repeatable.add(c["name"])
            seen.add(key)
            if ask is not None and c["name"] == ask.tool_name:
                tool, params = ask.parse(args)
                self.asked.update(f"{tool}.{p}" for p in params)
            for k, v in args.items():
                sh = shape_of(v)
                self.by_param[(c["name"], k)][sh] += 1
                if sh in TIME_SHAPES:
                    self.global_time[sh] += 1
                if isinstance(v, str) and 0 < len(v) <= 40:
                    self.values[(c["name"], k)][v] += 1

    def param_seen(self, tool: str, param: str) -> int:
        return sum(self.by_param.get((tool, param), Counter()).values())

    def names(self, tool: str, param: str) -> set[str]:
        """Values of a param that names something stored (a contact, a group): frequent, few distinct, capitalised."""
        c = self.values.get((tool, param))
        if not c:
            return self.name_sets.get((tool, param), set())
        n = sum(c.values())
        upper = sum(k for v, k in c.items() if v[:1].isupper())
        if n < 20 or len(c) > 0.5 * n or upper < 0.9 * n:
            return set()
        return set(c)

    def expected(self, tool: str, param: str, schema: dict | None = None) -> str | None:
        c = self.by_param.get((tool, param))
        if c:
            sh, _ = c.most_common(1)[0]
            return sh if sh in TIME_SHAPES else None
        schema = schema or {}
        fmt = schema.get("format")
        if fmt in ("date", "time"):
            return fmt
        desc = str(schema.get("description") or "")
        clock = re.search(r"hh|giờ|thời điểm|time|\d{2}:\d{2}", desc, re.I)
        if fmt != "date-time" and not clock and (
                re.search(r"yyyy-mm-dd", desc, re.I)
                or (re.search(r"date|ngay", param, re.I) and not re.search(r"time|gio", param, re.I))):
            return "date"
        if fmt == "date-time" or clock or TIME_PARAM_RX.search(param):
            # unseen time parameter: the house date-time style, never the date-only one
            dts = Counter({k: n for k, n in self.global_time.items() if k in DATETIME_SHAPES})
            if dts:
                return dts.most_common(1)[0][0]
        return None

    def to_json(self) -> dict:
        keys = set(self.values) | set(self.name_sets)
        names = {f"{t}::{p}": sorted(self.names(t, p)) for t, p in sorted(keys) if self.names(t, p)}
        return {"by_param": {f"{t}::{p}": dict(c) for (t, p), c in self.by_param.items()},
                "global_time": dict(self.global_time), "names": names, "tool_calls": dict(self.tool_calls),
                "repeatable": sorted(self.repeatable), "asked": sorted(self.asked)}

    @classmethod
    def from_json(cls, d: dict) -> "FormatBook":
        fb = cls()
        for k, c in (d.get("by_param") or {}).items():
            t, p = k.split("::", 1)
            fb.by_param[(t, p)] = Counter(c)
        fb.global_time = Counter(d.get("global_time") or {})
        for k, v in (d.get("names") or {}).items():
            t, p = k.split("::", 1)
            fb.name_sets[(t, p)] = set(v)
        fb.tool_calls = Counter(d.get("tool_calls") or {})
        fb.repeatable = set(d.get("repeatable") or [])
        fb.asked = set(d.get("asked") or [])
        return fb
