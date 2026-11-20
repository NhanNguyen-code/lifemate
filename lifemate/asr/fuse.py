"""Fuse two ASR hypotheses of the final request: the primary (PhoWhisper) and a secondary (gipformer).

Where the two disagree, the primary's words stay unless the LLM finds the sentence with the secondary's words more
likely, by more than `delta` nats (+ `beta` per extra token). On val, the secondary is right on ~40% of the content
words the primary gets wrong. Only the sample's own data is used: its two transcripts and its history.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ..data import Sample, is_user_turn
from ..textnorm import normalize

PARAMS_FILE = "fuse.json"          # in assets/: {"delta": float, "beta": float}
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def _key(tok: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFC", tok).lower())


# spellings the agent reads the same way: not worth an LM call
_SAME = {"ngàn": "nghìn", "dùm": "giùm", "nhé": "nha", "nhá": "nha", "nhe": "nha", "nghen": "nha", "hen": "nha",
         "hông": "không", "vô": "vào", "rùi": "rồi", "zalo": "zalo", "ok": "ok", "oke": "ok", "okay": "ok"}


def _norm(text: str) -> str:
    words = re.findall(r"\w+", normalize(unicodedata.normalize("NFC", text)).text.lower())
    return " ".join(_SAME.get(w, w) for w in words)


def _align(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


@dataclass
class Region:
    start: int                      # primary tokens [start, end) are replaced
    end: int
    words: list[str] = field(default_factory=list)   # by these secondary tokens


def regions(primary: str, secondary: str, max_diff: float = 0.5) -> tuple[list[str], list[Region]]:
    """Primary tokens and the spans where the secondary says something else (format-only differences such as
    "50" / "năm mươi" or casing are not spans). No spans when the two are too different to align."""
    at, bt = (primary or "").split(), (secondary or "").split()
    if not at or not bt:
        return at, []
    ak, bk = [_key(t) for t in at], [_key(t) for t in bt]
    runs: list[Region] = []
    cur: Region | None = None
    pos = 0                                           # next primary index
    for i, j in _align(ak, bk):
        same = i is not None and j is not None and ak[i] == bk[j]
        if same:
            if cur is not None:
                runs.append(cur)
                cur = None
            pos = i + 1
            continue
        if cur is None:
            cur = Region(pos, pos)
        if i is not None:
            cur.end = i + 1
            pos = i + 1
        if j is not None:
            cur.words.append(bt[j])
    if cur is not None:
        runs.append(cur)
    out = [r for r in runs if _norm(" ".join(at[r.start:r.end])) != _norm(" ".join(r.words))]
    changed = sum((r.end - r.start) + len(r.words) for r in out)
    if changed > max_diff * (len(at) + len(bt)):
        return at, []
    return at, out


def apply(tokens: list[str], spans: list[Region], chosen: list[bool]) -> str:
    out, i = [], 0
    for r, use in sorted(zip(spans, chosen), key=lambda x: (x[0].start, x[0].end)):
        out += tokens[i:r.start]
        out += r.words if use else tokens[r.start:r.end]
        i = max(i, r.end)
    out += tokens[i:]
    return " ".join(out)


def variants(tokens: list[str], spans: list[Region]) -> list[str]:
    """[primary, primary with span 0 replaced, primary with span 1 replaced, ...]"""
    n = len(spans)
    return [apply(tokens, spans, [False] * n)] + [apply(tokens, spans, [k == j for k in range(n)]) for j in range(n)]


def lm_prefix(sample: Sample) -> str:
    """The assistant's last reply gives the words of a short answer ("Zalo nhé em") a context."""
    asst = [t for t in sample.history if not is_user_turn(t)]
    last = " ".join(str(asst[-1].get("text") or "").split())[-300:] if asst else ""
    return (f"Trợ lý: {last}\nNgười dùng:" if last else "Người dùng nói với trợ lý ảo:")


def choose(lps: list[float], lens: list[int], delta: float, beta: float) -> list[bool]:
    """lps/lens: log-prob and token count of variants() (index 0 = primary)."""
    base, n0 = lps[0], lens[0]
    return [(lp - base) + beta * (n - n0) > delta for lp, n in zip(lps[1:], lens[1:])]


def read_params(*folders) -> dict | None:
    for f in folders:
        if f and (Path(f) / PARAMS_FILE).exists():
            return json.loads((Path(f) / PARAMS_FILE).read_text(encoding="utf-8"))
    return None


class Fuser:
    """LM-scored fusion with a loaded backend (anything with .score(ScoreRequest list) and .tok)."""

    def __init__(self, backend, delta: float = 1.0, beta: float = 0.0, max_spans: int = 8):
        self.backend = backend
        self.delta, self.beta, self.max_spans = delta, beta, max_spans

    def score(self, items: list[tuple[str, list[str]]]) -> list[tuple[list[float], list[int]]]:
        """items: (prefix, texts) -> per item the log-probs and token counts of " " + text after the prefix."""
        from ..llm import ScoreRequest
        reqs = [ScoreRequest(None, p, [" " + t for t in texts]) for p, texts in items]
        lps = self.backend.score(reqs) if reqs else []
        tok = self.backend.tok
        out = []
        for (p, texts), row in zip(items, lps):
            base = len(tok.encode(p, add_special_tokens=False))
            lens = [len(tok.encode(p + " " + t, add_special_tokens=False)) - base for t in texts]
            out.append((list(row), lens))
        return out

    def plan(self, sample: Sample, primary: str, secondary: str):
        tokens, spans = regions(primary, secondary)
        spans = spans[: self.max_spans]
        return tokens, spans, (variants(tokens, spans) if spans else [])

    def fuse_many(self, rows: list[tuple[Sample, str, str]]) -> list[str]:
        plans = [self.plan(s, a, b) for s, a, b in rows]
        todo = [k for k, p in enumerate(plans) if p[1]]
        scored = self.score([(lm_prefix(rows[k][0]), plans[k][2]) for k in todo])
        out = [a for _, a, _ in rows]
        for k, (lps, lens) in zip(todo, scored):
            tokens, spans, _ = plans[k]
            out[k] = apply(tokens, spans, choose(lps, lens, self.delta, self.beta))
        return out


# ---------------------------------------------------------------- offline: score once, tune on val, apply
def score_samples(fuser: Fuser, samples: list[Sample], primary, secondary, batch: int = 256) -> dict:
    """{id: {"lps": [...], "lens": [...]}} for samples whose two transcripts disagree (primary/secondary:
    Transcripts)."""
    rows = [(s, primary.final(s) or "", secondary.final(s) or "") for s in samples
            if primary.has(s.id) and secondary.has(s.id)]
    plans = [(s, fuser.plan(s, a, b)) for s, a, b in rows]
    plans = [(s, p) for s, p in plans if p[1]]
    out: dict = {}
    for k in range(0, len(plans), batch):
        chunk = plans[k:k + batch]
        res = fuser.score([(lm_prefix(s), p[2]) for s, p in chunk])
        for (s, _), (lps, lens) in zip(chunk, res):
            out[s.id] = {"lps": lps, "lens": lens}
    return out


def fused_finals(samples: list[Sample], primary, secondary, scores: dict, delta: float, beta: float,
                 max_spans: int = 8) -> dict[str, str]:
    out = {}
    for s in samples:
        if not primary.has(s.id):
            continue
        a = primary.final(s) or ""
        sc = scores.get(s.id)
        if sc is None or not secondary.has(s.id):
            out[s.id] = a
            continue
        tokens, spans = regions(a, secondary.final(s) or "")
        spans = spans[:max_spans]
        if len(sc["lps"]) != len(spans) + 1:          # scores from another primary/secondary pair
            out[s.id] = a
            continue
        out[s.id] = apply(tokens, spans, choose(sc["lps"], sc["lens"], delta, beta))
    return out


def final_wer(samples: list[Sample], finals: dict[str, str]) -> float:
    from .metrics import _words, edits
    e = n = 0
    for s in samples:
        ref = s.transcripts()[0]
        if ref is None or s.id not in finals:
            continue
        r = _words(normalize(ref).text)
        e += edits(r, _words(normalize(finals[s.id]).text))
        n += len(r)
    return 100.0 * e / max(1, n)


DELTAS = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, float("inf"))
BETAS = (-0.5, -0.25, 0.0, 0.25, 0.5)


def tune(samples: list[Sample], primary, secondary, scores: dict) -> tuple[dict, list[tuple]]:
    """delta/beta with the lowest WER of the final request; ties go to the larger delta (fewer changes)."""
    table = []
    for d in DELTAS:
        for b in (BETAS if d != float("inf") else (0.0,)):
            w = final_wer(samples, fused_finals(samples, primary, secondary, scores, d, b))
            table.append((round(w, 3), -d, b, d))
    table.sort()
    w, _, b, d = table[0]
    return {"delta": d, "beta": b, "wer_final": w}, [(d_, b_, w_) for w_, _, b_, d_ in table]
