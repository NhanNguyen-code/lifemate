"""Days the user names by weekday or by day and month, worked out from current_time.

The prompt has a table of this week and next week, but the model still slips on "thứ bảy tuần sau" now and then; a
ready date next to the request is harder to misread. Only phrases with one reading get a hint: a bare "thứ ba" (this
week or next?) and a bare "ngày 5" (which month?) are left to the model.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

from .textnorm import normalize
from .timeutil import now_from_context

_WD = {"hai": 0, "ba": 1, "tư": 2, "bốn": 2, "năm": 3, "sáu": 4, "bảy": 5, "bẩy": 5,
       "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5}
_WD_NAME = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
_WEEKDAY = re.compile(r"\b(?:thứ\s+(hai|ba|tư|bốn|năm|sáu|bảy|bẩy|[2-7])|(chủ\s+nhật))\s+"
                      r"(tuần\s+sau\s+nữa|tuần\s+(?:sau|tới)|tuần\s+này|này|tới)(?!\w)")
_SAID = r"(?:(?:ngày|mùng|mồng|hôm)\s+)"
# "ngày 11 tháng 12", "mùng 4 tháng 12", "thứ tư 14 tháng 10"; "11/12" only after a day word (8/10 may be a score)
_DAY_MONTH = re.compile(r"(?<!\w)(" + _SAID + r"*)(\d{1,2})\s+tháng\s+(\d{1,2})"
                        r"(?!\s*(?:[/:\d]|lần|người|triệu|nghìn|ngàn|ngày|tuần|năm|giờ|tiếng|phút|%))")
_DAY_SLASH = re.compile(r"(?<!\w)" + _SAID + r"+(\d{1,2})\s*/\s*(\d{1,2})(?!\s*[/:\d])")
_DAY_REL_MONTH = re.compile(r"(?<!\w)" + _SAID + r"*(\d{1,2})\s+tháng\s+(này|sau|tới)(?!\w)")
# "trong 3 tháng 2 lần" is a span of time, "cái thứ hai này" a rank
_SPAN = {"trong", "mỗi", "suốt", "khoảng", "hơn", "gần", "sau", "được", "cả", "đủ", "mấy", "vài", "qua"}
_RANK = {"cái", "chiếc", "người", "lần", "bài", "tờ", "món", "vé", "chuyến", "câu", "điều", "việc", "bước", "cuốn",
         "quyển", "tập", "đứa", "con", "số", "hàng", "tầng", "lớp"}


def _before(t: str, pos: int) -> str:
    w = t[:pos].split()
    return w[-1] if w else ""


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def resolve(text: str, ctx: dict) -> list[tuple[str, date]]:
    """[(phrase as said, date)] in order of appearance."""
    if not text:
        return []
    try:
        today = now_from_context(ctx).date()
    except Exception:
        return []
    t = unicodedata.normalize("NFC", normalize(text).text).lower()
    monday = today - timedelta(days=today.weekday())
    found: list[tuple[int, str, date]] = []
    for m in _WEEKDAY.finditer(t):
        wd = 6 if m.group(2) else _WD[m.group(1)]
        q = " ".join(m.group(3).split())
        if q == "này" and _before(t, m.start()) in _RANK:
            continue
        if q == "tuần sau nữa":
            d = monday + timedelta(days=14 + wd)
        elif q in ("tuần sau", "tuần tới"):
            d = monday + timedelta(days=7 + wd)
        elif q == "tuần này":
            d = monday + timedelta(days=wd)
        else:                                   # "thứ bảy này", "thứ ba tới": the next such day, today included for "này"
            ahead = (wd - today.weekday()) % 7
            if q == "tới" and ahead == 0:
                ahead = 7
            d = today + timedelta(days=ahead)
        found.append((m.start(), f"{_WD_NAME[wd]} {q}", d))
    for m in _DAY_REL_MONTH.finditer(t):
        day, q = int(m.group(1)), m.group(2)
        y, mo = today.year, today.month
        if q != "này":
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        d = _safe(y, mo, day)
        if d:
            found.append((m.start(), f"{day} tháng {q}", d))
    pairs = [(m.start(), int(m.group(2)), int(m.group(3))) for m in _DAY_MONTH.finditer(t)
             if m.group(1) or _before(t, m.start()) not in _SPAN]
    pairs += [(m.start(), int(m.group(1)), int(m.group(2))) for m in _DAY_SLASH.finditer(t)]
    for pos, day, mo in pairs:
        d = _safe(today.year, mo, day) if 1 <= mo <= 12 else None
        if d is None:
            continue
        if (today - d).days > 180:              # "5 tháng 1" said in November is next January
            d = _safe(today.year + 1, mo, day) or d
        found.append((pos, f"{day}/{mo}", d))
    out, seen = [], set()
    for _, phrase, d in sorted(found, key=lambda x: x[0]):
        if (phrase, d) not in seen:
            seen.add((phrase, d))
            out.append((phrase, d))
    return out


def hints(text: str, ctx: dict) -> list[str]:
    return [f"{phrase} = {d.isoformat()} ({_WD_NAME[d.weekday()]})" for phrase, d in resolve(text, ctx)]
