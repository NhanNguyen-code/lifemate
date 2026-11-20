"""SFT conversations from gold trajectories, built with the same prompt code as inference."""
from __future__ import annotations

import copy
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .actions import call_to_action, finish
from .augment import (add_distractors, drop_slot_to_ask, make_rename, remove_app_to_oos,
                      rename_catalog_data, rename_sample_raw, shuffle_installed)
from .catalog import Catalog
from .data import Sample, Transcripts, gold_category, texts_for
from .fewshot import FewShotPool, usable_apps
from .prompt import PromptConfig, assistant_message, build_messages, result_message


@dataclass
class SFTConfig:
    p_asr: float = 0.5
    n_aug: int = 1
    p_rename: float = 0.6
    p_tool_rename: float = 0.5
    p_param_rename: float = 0.0
    p_distractor: float = 0.3
    p_remove_oos: float = 0.08
    p_drop_ask: float = 0.0
    p_confuse: float = 0.15
    p_fewshot_other_app: float = 0.0   # share of rows whose examples avoid the sample's own apps (like a new app)
    n_aug_synth: int = 0               # extra augmented copies of each synthetic sample (`lifemate synth`)
    shuffle: bool = True
    seed: int = 0

    @classmethod
    def from_json(cls, d: dict | None) -> "SFTConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


# words the fine-tuned PhoWhisper mixes up on val; the label stays, so the model learns to read the context
CONFUSABLE = {"nhắc": "nhắn", "nhắn": "nhắc"}
_CONF_RX = re.compile(r"\b(" + "|".join(CONFUSABLE) + r")\b", re.I)


def confuse_words(text: str) -> str:
    def sub(m):
        w = m.group(0)
        r = CONFUSABLE[w.lower()]
        return r[0].upper() + r[1:] if w[0].isupper() else r
    return _CONF_RX.sub(sub, text, count=1)


OOS_THOUGHT = "Không ứng dụng nào trong danh sách làm được yêu cầu này nên dừng, không gọi tool."


def trajectory_messages(sample: Sample, cat: Catalog, final: str, hist: list[str], fewshot: str,
                        pcfg: PromptConfig, thoughts: list[str | None] | None = None) -> list[dict]:
    """thoughts[i] goes before gold step i; thoughts[len(gold)] before the closing finish."""
    tools = cat.allowed_tools(sample.installed_tool_names())
    msgs = build_messages(sample, cat, final, hist, fewshot, pcfg, tools)
    gold = sample.gold_calls or []
    results = sample.gold_results
    th = list(thoughts or [])
    for i, c in enumerate(gold):
        a = call_to_action(c, cat)
        if i < len(th) and th[i]:
            a = {"thought": th[i], **a}
        msgs.append(assistant_message(a))
        if a["type"] == "ask":
            return msgs
        res = results[i] if i < len(results) else {"ok": True}
        msgs.append(result_message(c["name"], res, pcfg))
    msgs.append(assistant_message(finish(th[len(gold)] if len(gold) < len(th) else None)))
    return msgs


def load_thoughts(paths) -> dict[str, list[str | None]]:
    """rationale / synth output files -> {sample id: [thought for step 0, 1, ...]}."""
    from .data import read_jsonl
    by: dict[str, dict[int, str]] = {}
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        for r in read_jsonl(path):
            if r.get("thought"):
                by.setdefault(r["id"], {})[int(r["step"])] = r["thought"]
    return {k: [v.get(i) for i in range(max(v) + 1)] for k, v in by.items()}


def _drop_apps(raw: dict, apps: set[str]) -> dict:
    if not apps:
        return raw
    r = copy.deepcopy(raw)
    inst = r["context"]["installed_apps"]
    for a in list(inst):
        if a in apps:
            inst.pop(a)
    return r


def build_rows(samples: list[Sample], catalog: Catalog, transcripts: Transcripts | None,
               pool: FewShotPool | None, pcfg: PromptConfig, scfg: SFTConfig,
               exclude_apps: set[str] | None = None, exclude_fewshot: set[str] | None = None,
               thoughts: dict[str, list] | None = None) -> tuple[list[dict], dict]:
    exclude_apps = exclude_apps or set()
    exclude_fewshot = exclude_fewshot or set()
    ask = catalog.ask_format()
    rows, stat = [], Counter()
    for s in samples:
        if s.gold_calls is None:
            continue
        n_aug = scfg.n_aug_synth if s.origin else scfg.n_aug
        for v in range(1 + max(0, n_aug)):
            rng = random.Random(f"{scfg.seed}:{s.id}:{v}")
            raw = _drop_apps(s.raw, exclude_apps)      # held-out apps must not leak into training prompts
            cat = catalog
            tags = ["synth"] if s.origin else []
            forced_oracle = False
            if v > 0:
                if rng.random() < scfg.p_remove_oos:
                    r2 = remove_app_to_oos(raw, cat)
                    if r2 is not None:
                        raw, _ = r2, tags.append("to_oos")
                if "to_oos" not in tags and rng.random() < scfg.p_drop_ask:
                    r2 = drop_slot_to_ask(raw, cat, ask)
                    if r2 is not None:
                        raw, forced_oracle = r2, True
                        tags.append("to_ask")
                if rng.random() < scfg.p_distractor:
                    r2 = add_distractors(raw, cat, rng, exclude=exclude_apps)
                    if r2 is not None:
                        raw = r2
                        tags.append("distractor")
                if rng.random() < scfg.p_rename:
                    apps = list(raw["context"]["installed_apps"].keys())
                    rn = make_rename(cat, apps, rng, 1.0, scfg.p_tool_rename, scfg.p_param_rename)
                    cat = Catalog(rename_catalog_data(cat.data, rn))
                    raw = rename_sample_raw(raw, rn, ask)
                    tags.append("rename")
                if scfg.shuffle:
                    raw = shuffle_installed(raw, rng)
            sv = Sample(raw)
            # untuned-ASR text so the model gets used to real recognition errors
            use_asr = (not forced_oracle and transcripts is not None and transcripts.has(s.id)
                       and rng.random() < scfg.p_asr)
            final, hist = texts_for(sv, transcripts if use_asr else None, prefer="asr" if use_asr else "oracle")
            if use_asr:
                tags.append("asr")
            if final and rng.random() < scfg.p_confuse:
                swapped = confuse_words(final)
                if swapped != final:
                    final = swapped
                    tags.append("confuse")
            fs = ""
            if pool is not None and pcfg.fewshot_k > 0:
                allowed = usable_apps(s.installed_apps, catalog) if pcfg.fewshot_installed_only else None
                avoid = None
                if scfg.p_fewshot_other_app > 0 and s.gold_calls and rng.random() < scfg.p_fewshot_other_app:
                    avoid = {c["name"].split(".")[0] for c in s.gold_calls} - usable_apps((), catalog)
                    tags.append("fewshot_other_app")
                ex = {s.id} | exclude_fewshot | ({s.origin} if s.origin else set())
                fs = pool.block_for(final, pcfg.fewshot_k, exclude=ex, allowed_apps=allowed, avoid_apps=avoid)
            th = None
            if thoughts is not None:
                if not (sv.gold_calls or []):
                    # nothing to call: the teacher's reason for the original sample, the fixed one after an app
                    # was removed (or when the teacher wrote none)
                    th = (None if "to_oos" in tags else thoughts.get(s.id)) or [OOS_THOUGHT]
                elif "to_ask" not in tags and len(sv.gold_calls) == len(s.gold_calls):
                    th = thoughts.get(s.id)
                    ends_ask = call_to_action(s.gold_calls[-1], catalog)["type"] == "ask"
                    need = len(s.gold_calls) + (0 if ends_ask else 1)
                    if th is not None and (len(th) < need or not all(th[:need])):
                        th = None                  # a step without one (prompt too long for the teacher)
                        stat["thought_incomplete"] += 1
                if th:
                    tags.append("thought")
            msgs = trajectory_messages(sv, cat, final, hist, fs, pcfg, th)
            cat_name = gold_category(sv, cat.meta_tools)
            rows.append({"messages": msgs, "meta": {"id": s.id, "variant": v, "aug": tags, "category": cat_name}})
            stat[cat_name] += 1
            for t in tags:
                stat["aug:" + t] += 1
    stat["rows"] = len(rows)
    return rows, dict(stat)
