"""Prompt text for Whisper on one user turn: the assistant's last reply plus the names in the tool calls and results
before that turn. All of it comes from the sample's own history (input data), nothing from the environment.

Names the user says again ("Grace Nakamura", "Nexora Fiber") or a short answer to the assistant's question
("Zalo nhé em") are what ASR gets wrong most often when it hears them without this context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..data import Sample, is_user_turn

MAX_PROMPT_CHARS = 360
MAX_PROMPT_TOKENS = 190          # Whisper keeps at most 223 prompt tokens; 448 in all, the transcript needs room
SETTINGS_FILE = "lifemate_asr.json"   # next to an ASR adapter / merged model: {"prompt_final": bool, "beams": int}

_ID = re.compile(r"^[A-Za-z]{1,16}([_-][0-9A-Za-z]+)+$")            # c_01, evt_02, chat-3
_NUMERIC = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|[+\d][\d\s.,:/+-]*$)")
_SKIP_KEYS = {"id", "ok", "error", "status", "currency", "unit", "url", "link", "email", "emails", "phone",
              "phone_number", "image", "icon", "code", "hash", "token", "otp"}


def _keep(s: str) -> bool:
    if not 2 <= len(s) <= 40 or not any(c.isalpha() for c in s):
        return False
    if _ID.match(s) or _NUMERIC.match(s) or "@" in s or "://" in s:
        return False
    if s.isascii() and s == s.lower():            # tool names and enum values: "contacts.search", "pending"
        return False
    return len(s.split()) <= 6                    # longer strings are messages; the reply text covers those


def _collect(obj, out: list[str], depth: int = 0):
    if depth > 8 or len(out) >= 300:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            k = str(k).lower()
            if k in _SKIP_KEYS or k.endswith(("_id", "_ids")):
                continue
            _collect(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out, depth + 1)
    elif isinstance(obj, str):
        s = " ".join(obj.split())
        if _keep(s):
            out.append(s)


def turns_before(sample: Sample, user_index: int | None = None) -> list[dict]:
    """History turns before user turn `user_index` (counted among user turns); None = before the final request."""
    hist = sample.history
    if user_index is None:
        return hist
    seen = -1
    for i, t in enumerate(hist):
        if is_user_turn(t):
            seen += 1
            if seen == user_index:
                return hist[:i]
    return hist


def prompt_text(turns: list[dict], max_chars: int = MAX_PROMPT_CHARS) -> str:
    """"name, name, ... <last assistant reply>", within max_chars; empty when no assistant turn came before."""
    asst = [t for t in turns if not is_user_turn(t)]
    if not asst:
        return ""
    last = " ".join(str(asst[-1].get("text") or "").split())
    if len(last) > max_chars // 2:                # keep the end of a long reply, from a word boundary
        last = last[-(max_chars // 2):].split(" ", 1)[-1]
    names: list[str] = []
    for t in reversed(asst):                      # most recent turn first
        _collect(t.get("tool_calls"), names)
        _collect(t.get("tool_results"), names)
    low = last.casefold()
    seen: set[str] = set()
    picked: list[str] = []
    room = max_chars - len(last) - 2
    for n in names:
        f = n.casefold()
        if f in seen or f in low:
            continue
        if len(n) + 2 > room:
            break
        seen.add(f)
        picked.append(n)
        room -= len(n) + 2
    parts = ([", ".join(picked) + "."] if picked else []) + ([last] if last else [])
    return " ".join(parts)


def prompt_for(sample: Sample, user_index: int | None = None, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return prompt_text(turns_before(sample, user_index), max_chars)


def clip_prompt_ids(ids: list[int], max_tokens: int = MAX_PROMPT_TOKENS) -> list[int]:
    """Keep <|startofprev|> and the last tokens: the end of the prompt (the reply) matters most."""
    ids = [int(x) for x in ids]
    if len(ids) <= max_tokens:
        return ids
    return ids[:1] + ids[len(ids) - (max_tokens - 1):]


def read_settings(*folders) -> dict:
    for f in folders:
        if f and (Path(f) / SETTINGS_FILE).exists():
            try:
                return json.loads((Path(f) / SETTINGS_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def write_settings(folder, **kw) -> dict:
    p = Path(folder) / SETTINGS_FILE
    cur = read_settings(folder)
    cur.update(kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    return cur
