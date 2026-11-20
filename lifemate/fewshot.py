"""Few-shot retrieval from train with char n-gram TF-IDF (pure python)."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .data import Sample, read_jsonl, write_jsonl
from .textnorm import strip_diacritics

HEADER = "# Ví dụ đã làm đúng (chỉ tham khảo cách làm; ID trong ví dụ không dùng được)"


def _grams(text: str, lo: int = 3, hi: int = 5) -> Counter:
    t = " " + " ".join(strip_diacritics(text.lower()).split()) + " "
    c: Counter = Counter()
    for n in range(lo, hi + 1):
        for i in range(len(t) - n + 1):
            c[t[i:i + n]] += 1
    return c


def render_trace(calls: list[dict], catalog) -> str:
    if not calls:
        return "(không gọi tool nào: không có ứng dụng phù hợp)"
    parts = []
    for c in calls:
        args = dict(c.get("arguments") or {})
        spec = catalog.tools.get(c["name"]) if catalog is not None else None
        if spec is not None:
            for k in spec.ref_keys():
                if k in args:
                    args[k] = "<ID>"
        parts.append(f"{c['name']} {json.dumps(args, ensure_ascii=False)}")
    return " → ".join(parts)


class FewShotPool:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self._vecs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._post: dict[str, list[tuple[int, float]]] = {}
        self._build()

    def _build(self):
        df: Counter = Counter()
        grams = []
        for e in self.entries:
            g = _grams(e["text"])
            grams.append(g)
            df.update(g.keys())
        n = max(1, len(self.entries))
        self._idf = {k: math.log((1 + n) / (1 + v)) + 1.0 for k, v in df.items()}
        self._vecs = [self._vec(g) for g in grams]
        # inverted index: only score entries that share an n-gram with the query
        post: dict[str, list[tuple[int, float]]] = {}
        for i, v in enumerate(self._vecs):
            for g, w in v.items():
                post.setdefault(g, []).append((i, w))
        self._post = post

    def _vec(self, g: Counter) -> dict[str, float]:
        v = {k: (1 + math.log(c)) * self._idf.get(k, 0.0) for k, c in g.items() if k in self._idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    @classmethod
    def from_samples(cls, samples: list[Sample], catalog) -> "FewShotPool":
        entries = []
        for s in samples:
            if s.gold_calls is None or s.audio_text is None:
                continue
            entries.append({"id": s.id, "text": s.audio_text,
                            "trace": render_trace(s.gold_calls, catalog),
                            "apps": sorted({c["name"].split(".")[0] for c in s.gold_calls})})
        return cls(entries)

    def save(self, path: str | Path):
        write_jsonl(path, self.entries)

    @classmethod
    def load(cls, path: str | Path) -> "FewShotPool":
        return cls(read_jsonl(path))

    def search(self, text: str, k: int, exclude: set[str] | None = None, allowed_apps: set[str] | None = None,
               avoid_apps: set[str] | None = None) -> list[dict]:
        """allowed_apps: only examples whose apps all are in it; avoid_apps: no example that uses one of them."""
        if k <= 0 or not self.entries:
            return []
        q = self._vec(_grams(text))
        exclude = exclude or set()
        acc: dict[int, float] = {}
        for g, w in q.items():
            for i, x in self._post.get(g, ()):
                acc[i] = acc.get(i, 0.0) + w * x

        def ok(e: dict) -> bool:
            if e["id"] in exclude:
                return False
            apps = set(e.get("apps") or [])
            if allowed_apps is not None and not apps <= allowed_apps:
                return False
            return not (avoid_apps and apps & avoid_apps)

        scored = [(s, self.entries[i]["id"], i) for i, s in acc.items() if ok(self.entries[i])]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.entries[i] for _, _, i in scored[:k]]

    @staticmethod
    def render(entries: list[dict]) -> str:
        if not entries:
            return ""
        lines = [HEADER]
        for e in entries:
            lines.append(f'- "{e["text"]}" ⇒ {e["trace"]}')
        return "\n".join(lines)

    def block_for(self, text: str, k: int, exclude: set[str] | None = None, allowed_apps: set[str] | None = None,
                  avoid_apps: set[str] | None = None) -> str:
        return self.render(self.search(text, k, exclude, allowed_apps, avoid_apps))


def usable_apps(installed, catalog) -> set[str]:
    """Apps a few-shot example may use: the installed ones plus the system (ask) tools."""
    return set(installed) | {n.split(".")[0] for n in catalog.meta_tools}
