"""Inverse text normalization for Vietnamese ASR output: money, times, percents, digit strings.

Only spans that are clearly numbers get converted, so "năm nay" or "ba mẹ" stay untouched.
The original text is kept; the prompt shows the normalized form and hints next to it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DIGIT = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5,
         "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
UNIT_AFTER_TENS = {**DIGIT, "mốt": 1, "lăm": 5, "nhăm": 5}
UNIT_AFTER_TENS.pop("không")
SCALES = {"nghìn": 1000, "ngàn": 1000, "k": 1000, "triệu": 1_000_000, "củ": 1_000_000,
          "tỷ": 1_000_000_000, "tỉ": 1_000_000_000}
MONEY_UNITS = {"đồng", "vnđ", "vnd", "đ"}
MONEY_CUES = {"chuyển", "gửi", "nạp", "trả", "rút", "thanh", "toán", "giá", "hết", "tiền", "khoản", "vay", "mượn"}
COUNT_UNITS = {"phút", "giây", "tiếng", "ngày", "tháng", "tuần", "lần", "người", "cái", "chiếc", "độ",
               "km", "mét", "phần", "suất", "vé", "chai", "ly", "cốc", "phòng", "tầng", "số", "tuổi",
               "bàn", "ghế", "đơn", "món", "kg", "lít"}
PERIODS = {"sáng", "trưa", "chiều", "tối", "đêm", "khuya"}
NOT_NUMBER_BEFORE = {"thứ"}
# spoken tens without "mươi": "hai tư" = 24, "bốn lăm" = 45, "ba mốt" = 31
COLLOQ_UNIT = {"mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5}
SEPARATORS = {",", "-", ".", "–"}
SEQ_DIGIT = {**DIGIT, "lăm": 5, "nhăm": 5, "mốt": 1}
KEM = {"kém", "kẻo", "kế", "kem", "kèm"}     # ASR often writes "kém" as "kẻo" / "kế"

_TOKEN_RE = re.compile(r"\d{1,2}[hg:]\d{2}|\d+(?:[.,]\d+)*[a-zđ%]*|[^\W\d_]+|[^\w\s]", re.UNICODE)


@dataclass
class NormResult:
    text: str
    hints: list[str] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _is_digit(t: str) -> bool:
    return t in DIGIT


def _colloq_tens(toks: list[str], j: int) -> int | None:
    n = len(toks)
    if j + 1 >= n or toks[j] not in DIGIT or toks[j] == "tư" or DIGIT[toks[j]] < 2:
        return None
    d, u = DIGIT[toks[j]], toks[j + 1]
    if u in COLLOQ_UNIT:
        return d * 10 + COLLOQ_UNIT[u]
    # "hai sáu độ" = 26, but "hai ba ngày" is 2-3 days: only non-adjacent digits right before a unit
    if (u in DIGIT and u not in ("không", "một", "năm") and abs(DIGIT[u] - d) != 1
            and (toks[j], u) != ("năm", "bảy")):         # "năm bảy lần" = a few times
        nxt = toks[j + 2] if j + 2 < n else ""
        if nxt in COUNT_UNITS or nxt in SCALES or nxt in MONEY_UNITS or nxt in ("giờ", "phần", "%"):
            return d * 10 + DIGIT[u]
    return None


def _parse_lt1000(toks: list[str], i: int):
    n = len(toks)
    v, j, seen, mult, after_hundred = 0, i, False, False, False
    if j + 1 < n and _is_digit(toks[j]) and toks[j + 1] == "trăm":
        v += DIGIT[toks[j]] * 100
        j += 2
        seen = mult = after_hundred = True
        if j < n and toks[j] == "rưỡi":
            return v + 50, j + 1, True
        if j + 1 < n and toks[j] in ("linh", "lẻ") and _is_digit(toks[j + 1]):
            return v + DIGIT[toks[j + 1]], j + 2, True
    elif j < n and toks[j] == "trăm":
        v, j, seen, mult, after_hundred = 100, j + 1, True, True, True
    if j < n and toks[j] == "mười":
        v += 10
        j += 1
        seen = mult = True
        if j < n and toks[j] in UNIT_AFTER_TENS:
            v += UNIT_AFTER_TENS[toks[j]]
            j += 1
    elif j + 1 < n and _is_digit(toks[j]) and toks[j + 1] in ("mươi", "chục"):
        v += DIGIT[toks[j]] * 10
        j += 2
        seen = mult = True
        if toks[j - 1] == "mươi" and j < n and toks[j] in UNIT_AFTER_TENS:
            v += UNIT_AFTER_TENS[toks[j]]
            j += 1
    elif j < n and toks[j] == "chục" and not seen:
        v, j, seen, mult = 10, j + 1, True, True
    elif j < n and toks[j] == "hăm":                      # "hăm lăm" = 25
        v += 20
        j += 1
        seen = mult = True
        if j < n and toks[j] in UNIT_AFTER_TENS:
            v += UNIT_AFTER_TENS[toks[j]]
            j += 1
    elif j < n and _is_digit(toks[j]):
        c = _colloq_tens(toks, j)
        if c is not None:                                  # "một trăm hai chín" = 129
            v += c
            j += 2
            seen = mult = True
        elif after_hundred:
            nxt = toks[j + 1] if j + 1 < n else ""
            if nxt not in SCALES and not _is_digit(nxt):
                v += DIGIT[toks[j]] * 10
                j += 1
        elif not seen:
            v += DIGIT[toks[j]]
            j += 1
            seen = True
    if not seen:
        return None
    return v, j, mult


def _numeric_token(t: str):
    m = re.fullmatch(r"(\d{1,3}(?:\.\d{3})+)", t)
    if m:
        return int(t.replace(".", "")), ""
    m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+)", t)
    if m:
        return int(t.replace(",", "")), ""
    m = re.fullmatch(r"(\d+)[,.](\d+)", t)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}"), ""
    m = re.fullmatch(r"(\d+)(k|%|tr)?", t)
    if m:
        v = int(m.group(1))
        suf = m.group(2) or ""
        if suf == "k":
            return v * 1000, "k"
        if suf == "tr":
            return v * 1_000_000, "tr"
        return v, suf
    return None


def _parse_cardinal(toks: list[str], i: int):
    n = len(toks)
    total, j, mult, prev_scale = 0, i, False, None
    while j < n:
        r = _parse_lt1000(toks, j)
        if r is None:
            break
        v, j2, m = r
        mult = mult or m
        if j2 < n and toks[j2] in SCALES:
            scale = SCALES[toks[j2]]
            j2 += 1
            val = v * scale
            if j2 < n and toks[j2] == "rưỡi":
                val += scale // 2
                j2 += 1
            elif (j2 < n and _is_digit(toks[j2])
                  and (j2 + 1 >= n or (toks[j2 + 1] not in ("mươi", "chục", "trăm")
                                       and toks[j2 + 1] not in SCALES))):
                val += DIGIT[toks[j2]] * scale // 10      # "một triệu hai" = 1,200,000
                j2 += 1
            total += val
            j, mult, prev_scale = j2, True, scale
            continue
        if prev_scale and prev_scale >= 1000 and v < 1000:
            v *= prev_scale // 1000          # "hai triệu ba trăm" = 2,300,000
        total += v
        j = j2
        break
    if j == i:
        return None
    return total, j, mult


def _digit_run(toks: list[str], i: int):
    j = i
    while j < len(toks) and _is_digit(toks[j]) and toks[j] != "tư":
        j += 1
    if j - i >= 3:
        nxt = toks[j] if j < len(toks) else ""
        if nxt not in ("mươi", "trăm", "chục", "mười") and nxt not in SCALES:
            return "".join(str(DIGIT[t]) for t in toks[i:j]), j
    return None


def _spoken_digits(toks: list[str], i: int):
    # a long spoken run is one digit per word, even "chín lăm": "không chín lăm không năm ba sáu, ..."
    if toks[i] not in SEQ_DIGIT:
        return None
    digits, j, end = "", i, i
    while j < len(toks):
        if toks[j] in SEQ_DIGIT:
            digits += str(SEQ_DIGIT[toks[j]])
            j += 1
            end = j
        elif toks[j] in SEPARATORS and j + 1 < len(toks) and toks[j + 1] in SEQ_DIGIT:
            j += 1
        else:
            break
    if (toks[i] == "không" and len(digits) >= 7) or len(digits) >= 9:
        return digits, end
    return None


def _digit_piece(toks: list[str], j: int):
    if re.fullmatch(r"\d+", toks[j]):
        return toks[j], j + 1
    k = j
    while k < len(toks) and _is_digit(toks[k]) and toks[k] != "tư":
        k += 1
    nxt = toks[k] if k < len(toks) else ""
    if k == j or nxt in ("mươi", "trăm", "chục", "mười") or nxt in COLLOQ_UNIT or nxt in SCALES:
        return None
    return "".join(str(DIGIT[t]) for t in toks[j:k]), k


def _digit_sequence(toks: list[str], i: int):
    # phone / account numbers: keep leading zeros, join groups ASR splits ("0950 536 0367")
    first = _digit_piece(toks, i)
    if first is None:
        return None
    pieces, j = [first[0]], first[1]
    while j < len(toks):
        k = j + 1 if toks[j] in SEPARATORS else j
        p = _digit_piece(toks, k) if k < len(toks) else None
        if p is None:
            break
        pieces.append(p[0])
        j = p[1]
    digits = "".join(pieces)
    nxt = toks[j] if j < len(toks) else ""
    unit_next = nxt in COUNT_UNITS or nxt in SCALES or nxt in MONEY_UNITS or nxt == "giờ"
    if (len(pieces) >= 2 and len(digits) >= 7 and (digits.startswith("0") or len(digits) >= 9)
            and not unit_next):
        return digits, j
    if re.fullmatch(r"0\d{2,}", toks[i]):
        return toks[i], i + 1
    return None


def _clock(toks: list[str], hour: int, j: int, start: int):
    n = len(toks)
    j += 1
    minute = 0
    if j < n and toks[j] == "rưỡi":
        minute, j = 30, j + 1
    elif j < n and toks[j] in KEM:
        r = _parse_cardinal(toks, j + 1) if j + 1 < n else None
        if r is None and j + 1 < n and re.fullmatch(r"\d{1,2}", toks[j + 1]):
            r = (int(toks[j + 1]), j + 2, False)          # "8 giờ kém 15"
        if r and 0 < r[0] < 60:
            minute = (60 - int(r[0])) % 60
            hour = (hour - 1) % 24
            j = r[1]
            if j < n and toks[j] == "phút":
                j += 1
    elif j < n and toks[j] == "đúng":
        j += 1
    else:
        r = _parse_cardinal(toks, j) if j < n else None
        if r and 0 <= r[0] < 60:
            minute, j = int(r[0]), r[1]
            if j < n and toks[j] == "phút":
                j += 1
        else:
            m = re.fullmatch(r"\d{1,2}", toks[j]) if j < n else None
            if m and int(toks[j]) < 60:
                minute, j = int(toks[j]), j + 1
                if j < n and toks[j] == "phút":
                    j += 1
    period = toks[j] if j < n and toks[j] in PERIODS else None
    if period:
        j += 1
    else:
        period = _period_before(toks, start)
    hour = _apply_period(hour, period)
    return f"{hour:02d}:{minute:02d}", j


def _period_before(toks: list[str], i: int, window: int = 4) -> str | None:
    for k in range(i - 1, max(-1, i - 1 - window), -1):
        if toks[k] in PERIODS:
            return toks[k]
    return None


def _apply_period(hour: int, period: str | None) -> int:
    if period in ("chiều", "tối") and hour < 12:
        return hour + 12
    if period == "trưa" and hour < 6:
        return hour + 12
    if period in ("đêm", "khuya"):
        if hour == 12:
            return 0
        if 6 <= hour < 12:
            return hour + 12
    return hour % 24


def _fmt(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    return str(v)


def normalize(text: str) -> NormResult:
    toks = tokenize(text)
    out: list[str] = []
    hints: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        prev = toks[i - 1] if i > 0 else ""
        if prev in NOT_NUMBER_BEFORE and _is_digit(t):
            out.append(t)
            i += 1
            continue
        m = re.fullmatch(r"(\d{1,2})(?:h|g|:)(\d{2})?", t)
        if m and int(m.group(1)) < 25 and m.group(2) is None:
            hh, j = _clock(toks, int(m.group(1)), i, i)      # "8h kém 15", "8h rưỡi", "8h 45"
            out.append(hh)
            hints.append(f"giờ {hh}")
            i = j
            continue
        if m and int(m.group(1)) < 25:
            hour, minute = int(m.group(1)), int(m.group(2) or 0)
            j = i + 1
            period = toks[j] if j < n and toks[j] in PERIODS else None
            if period:
                j += 1
            else:
                period = _period_before(toks, i)
            hh = f"{_apply_period(hour, period):02d}:{minute:02d}"
            out.append(hh)
            hints.append(f"giờ {hh}")
            i = j
            continue
        seq = _spoken_digits(toks, i) or _digit_sequence(toks, i)
        if seq:
            digits, j = seq
            out.append(digits)
            hints.append(f"dãy số {digits}")
            i = j
            continue
        run = _digit_run(toks, i)
        if run:
            digits, j = run
            out.append(digits)
            hints.append(f"dãy số {digits}")
            i = j
            continue
        val, j, mult, numeric = None, i, False, False
        nt = _numeric_token(t)
        if nt is not None:
            val, suf = nt
            j, numeric = i + 1, True
            if suf == "%":
                out.append(f"{_fmt(val)}%")
                hints.append(f"phần trăm {_fmt(val)}")
                i = j
                continue
            if j < n and toks[j] in SCALES and not suf:
                val, j, mult = val * SCALES[toks[j]], j + 1, True
                if j < n and toks[j] == "rưỡi":
                    val, j = val + SCALES[toks[j - 1]] // 2, j + 1
            elif suf:
                mult = True
        elif t in ("mười", "trăm", "chục", "hăm") or _is_digit(t):
            if _is_digit(t) and i + 2 < n and toks[i + 1] == "phẩy" and _is_digit(toks[i + 2]):
                k = i + 2
                dec = ""
                while k < n and _is_digit(toks[k]):
                    dec += str(DIGIT[toks[k]])
                    k += 1
                v = float(f"{DIGIT[t]}.{dec}")
                if k < n and toks[k] in SCALES:
                    val, j, mult = v * SCALES[toks[k]], k + 1, True
                else:
                    val, j, mult = v, k, True
            else:
                r = _parse_cardinal(toks, i)
                if r:
                    val, j, mult = r
        if val is None:
            out.append(t)
            i += 1
            continue
        nxt = toks[j] if j < n else ""
        if (nxt == "rưỡi" and isinstance(val, int) and 0 < val < 25
                and ((j + 1 < n and toks[j + 1] in PERIODS) or prev in ("lúc", "vào"))):
            period = toks[j + 1] if j + 1 < n and toks[j + 1] in PERIODS else None
            hh = f"{_apply_period(val, period or _period_before(toks, i)):02d}:30"   # "tám rưỡi tối"
            out.append(hh)
            hints.append(f"giờ {hh}")
            i = j + 2 if period else j + 1
            continue
        if nxt == "giờ" and isinstance(val, (int, float)) and val < 25:
            hh, j2 = _clock(toks, int(val), j, i)
            out.append(hh)
            hints.append(f"giờ {hh}")
            i = j2
            continue
        if nxt == "phần" and j + 1 < n and toks[j + 1] == "trăm":
            out.append(f"{_fmt(val)}%")
            hints.append(f"phần trăm {_fmt(val)}")
            i = j + 2
            continue
        money = nxt in MONEY_UNITS or (mult and val >= 1000) or any(
            w in MONEY_CUES for w in toks[max(0, i - 3): i])
        # a bare digit word is only a number next to a unit / scale / cue word
        confident = numeric or mult or nxt in COUNT_UNITS or nxt in MONEY_UNITS
        if not confident and prev in ("ngày", "mùng", "mồng", "tháng", "lúc", "số"):
            confident = True
        if prev == "tiếng" and isinstance(val, int) and 0 < val < 60 and nxt not in COUNT_UNITS:
            out.append(f"{val} phút")                     # "1 tiếng 45" = 1h45
            hints.append(f"số {val} phút")
            i = j
            continue
        if not confident:
            out.extend(toks[i:j])
            i = j
            continue
        out.append(_fmt(val))
        if money and (val >= 1000 or nxt in MONEY_UNITS):
            hints.append(f"số tiền {_fmt(val)}")
        elif nxt in COUNT_UNITS or prev in ("ngày", "mùng", "mồng", "tháng", "số", "lúc"):
            hints.append(f"số {_fmt(val)}" + (f" {nxt}" if nxt in COUNT_UNITS else ""))
        i = j
    return NormResult(_detok(out), hints)


def clock_times(text: str | None) -> list[str]:
    return [h[4:] for h in normalize(text or "").hints if h.startswith("giờ ")]


def _detok(toks: list[str]) -> str:
    s = " ".join(toks)
    s = re.sub(r"\s+([,.!?;:%)])", r"\1", s)
    s = re.sub(r"([(])\s+", r"\1", s)
    return s


def strip_diacritics(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")
