"""Dataset access: split paths, samples, transcripts, and loading each split's own zaic_tools."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_ids(path: str | Path | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return list(obj.get("val_ids") or obj.get("ids") or [])
        return list(obj)
    return [x.strip() for x in txt.splitlines() if x.strip()]


@dataclass(frozen=True)
class SplitPaths:
    root: Path
    split: str

    @property
    def dir(self) -> Path:
        return self.root / self.split

    @property
    def jsonl(self) -> Path:
        return self.dir / f"{self.split}.jsonl"

    @property
    def catalog(self) -> Path:
        return self.dir / f"{self.split}_tools_catalog.json"

    @property
    def zaic_tools(self) -> Path:
        return self.dir / "zaic_tools"

    def audio(self, rel: str) -> Path:
        return self.dir / rel


def resolve_split(data: str | Path, split: str | None = None) -> SplitPaths:
    p = Path(data).resolve()
    if split and (p / split / f"{split}.jsonl").exists():
        return SplitPaths(p, split)
    if split and p.name == split and (p / f"{split}.jsonl").exists():
        return SplitPaths(p.parent, split)
    cands = [c.name[: -len("_tools_catalog.json")] for c in p.glob("*_tools_catalog.json")]
    cands = [c for c in cands if (p / f"{c}.jsonl").exists()]
    if split:
        cands = [c for c in cands if c == split] or cands
    if len(cands) == 1:
        name = cands[0]
        if p.name == name:
            return SplitPaths(p.parent, name)
        # ToolEnv wants <root>/<split>/...; a mounted dir such as /data can have any name
        tmp = Path(tempfile.mkdtemp(prefix="zroot_"))
        os.symlink(p, tmp / name, target_is_directory=True)
        return SplitPaths(tmp, name)
    raise FileNotFoundError(f"split '{split}' not found under {p} "
                            f"(expected <root>/<split>/<split>.jsonl or the split dir itself)")


_LOADED: dict[str, Any] = {}


def load_toolenv_class(sp: SplitPaths):
    key = str(sp.dir.resolve())
    if _LOADED.get("key") == key:
        return _LOADED["cls"]
    for m in list(sys.modules):
        if m == "zaic_tools" or m.startswith("zaic_tools."):
            del sys.modules[m]
    prev = _LOADED.get("path")
    if prev and prev in sys.path:
        sys.path.remove(prev)
    sys.path.insert(0, key)
    mod = importlib.import_module("zaic_tools")
    cls = getattr(mod, "ToolEnv")
    _LOADED.update(key=key, cls=cls, path=key)
    return cls


def make_env(sp: SplitPaths, sample_id: str):
    cls = load_toolenv_class(sp)
    return cls(sp.split, sample_id, root=str(sp.root))


@dataclass
class Sample:
    raw: dict

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def env_id(self) -> str:
        """Whose environment the sample runs on: itself, or the train sample a synthetic one was made from."""
        return self.raw.get("env_id") or self.raw["id"]

    @property
    def origin(self) -> str | None:
        """Source train sample of a synthetic sample (`lifemate synth`), else None."""
        return self.raw.get("origin")

    @property
    def audio(self) -> str:
        return self.raw.get("audio", "")

    @property
    def context(self) -> dict:
        return self.raw.get("context") or {}

    @property
    def installed_apps(self) -> dict:
        return self.context.get("installed_apps") or {}

    @property
    def history(self) -> list[dict]:
        return self.raw.get("history") or []

    @property
    def audio_text(self) -> str | None:
        return self.transcripts()[0]

    def transcripts(self) -> tuple[str | None, list[str | None]]:
        users = self.user_turns()
        n = len(users)
        hist = [_turn_text(t) for t in users]
        # README does not show the shape of audio_text, so accept str / list / dict
        at = self.raw.get("audio_text")
        final = None
        if isinstance(at, str):
            final = at
        elif isinstance(at, list):
            texts = [_as_text(x) for x in at]
            if texts:
                final = texts[-1]
                prev = texts[:-1][-n:] if n else []
                prev = [None] * (n - len(prev)) + prev
                hist = [h if h is not None else p for h, p in zip(hist, prev)]
        elif isinstance(at, dict):
            by_path = {str(k): _as_text(v) for k, v in at.items()}
            if self.audio in by_path or any(str(t.get("audio")) in by_path for t in users):
                final = by_path.get(self.audio)
                hist = [h if h is not None else by_path.get(str(t.get("audio"))) for h, t in zip(hist, users)]
            else:
                for k in ("final", "request", "current", "text", "audio_text"):
                    if _as_text(at.get(k)) is not None:
                        final = _as_text(at.get(k))
                        break
                hh = at.get("history")
                if isinstance(hh, list):
                    prev = [_as_text(x) for x in hh][-n:] if n else []
                    prev = [None] * (n - len(prev)) + prev
                    hist = [h if h is not None else p for h, p in zip(hist, prev)]
        else:
            final = _as_text(at)
        return final, hist

    @property
    def label(self) -> dict | None:
        return self.raw.get("label")

    @property
    def gold_calls(self) -> list[dict] | None:
        lab = self.label
        return None if lab is None else list(lab.get("tool_calls") or [])

    @property
    def gold_results(self) -> list[Any]:
        lab = self.label or {}
        return list(lab.get("tool_results") or [])

    def user_turns(self) -> list[dict]:
        return [t for t in self.history if is_user_turn(t)]

    def installed_tool_names(self) -> list[str]:
        out = []
        for app, spec in self.installed_apps.items():
            tools = spec.get("tools", []) if isinstance(spec, dict) else list(spec or [])
            for t in tools:
                name = t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else None)
                if not name:
                    continue
                out.append(name if "." in name else f"{app}.{name}")
        return out


def is_user_turn(turn: dict) -> bool:
    role = turn.get("role") or turn.get("speaker")
    if role:
        return str(role).lower() in ("user", "human", "nguoi_dung", "người dùng")
    return "audio" in turn


def _as_text(v) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("audio_text", "transcript", "text"):
            if isinstance(v.get(k), str):
                return v[k]
    return None


def _turn_text(turn: dict) -> str | None:
    for k in ("audio_text", "transcript", "text"):
        if isinstance(turn.get(k), str):
            return turn[k]
    return None


def load_samples(sp: SplitPaths, ids: Iterable[str] | None = None) -> list[Sample]:
    rows = [Sample(r) for r in read_jsonl(sp.jsonl)]
    if ids is not None:
        keep = set(ids)
        rows = [s for s in rows if s.id in keep]
    return rows


def gold_category(sample: Sample, meta_tools: set[str]) -> str:
    calls = sample.gold_calls
    if calls is None:
        return "unknown"
    if not calls:
        return "oos"
    if any(c["name"] in meta_tools for c in calls):
        return "ask"
    return "act"


@dataclass
class Transcripts:
    rows: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | list | None) -> "Transcripts":
        t = cls()
        for p in path if isinstance(path, (list, tuple)) else [path]:
            if p and Path(p).exists():
                for r in read_jsonl(p):
                    t.rows[r["id"]] = r          # later files win
        return t

    def has(self, sid: str) -> bool:
        return sid in self.rows

    def final(self, sample: Sample) -> str | None:
        r = self.rows.get(sample.id)
        return None if r is None else r.get("final")

    def history(self, sample: Sample) -> list[str] | None:
        r = self.rows.get(sample.id)
        return None if r is None else list(r.get("history") or [])


def oracle_texts(sample: Sample) -> tuple[str, list[str]]:
    final, hist = sample.transcripts()
    return final or "", [h or "" for h in hist]


def texts_for(sample: Sample, transcripts: Transcripts | None, prefer: str = "asr") -> tuple[str, list[str]]:
    if prefer == "asr" and transcripts is not None and transcripts.has(sample.id):
        f = transcripts.final(sample) or ""
        h = transcripts.history(sample) or []
        n = len(sample.user_turns())
        h = (h + [""] * n)[:n]
        return f, h
    if sample.audio_text is not None:
        return oracle_texts(sample)
    if prefer == "asr":
        raise KeyError(f"no ASR transcript for {sample.id}, run `python -m lifemate transcribe` first")
    return "", [""] * len(sample.user_turns())
