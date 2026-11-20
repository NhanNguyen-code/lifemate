"""Run a split: one ToolEnv per sample, write the submission and a trace."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, AgentConfig, Episode
from .catalog import Catalog
from .data import Sample, SplitPaths, Transcripts, make_env, texts_for, write_jsonl
from .fewshot import FewShotPool, usable_apps
from .guard import Guard, GuardConfig
from .llm import Backend
from .prompt import PromptConfig, build_messages
from .textnorm import clock_times
from .timeutil import FormatBook


@dataclass
class RunContext:
    sp: SplitPaths
    catalog: Catalog
    transcripts: Transcripts | None = None
    text_source: str = "asr"
    pool: FewShotPool | None = None
    formats: FormatBook | None = None
    prompt_cfg: PromptConfig = field(default_factory=PromptConfig)
    guard_cfg: GuardConfig = field(default_factory=GuardConfig)
    agent_cfg: AgentConfig = field(default_factory=AgentConfig)
    exclude_fewshot: set[str] = field(default_factory=set)
    exclude_origin: bool = False          # synthetic samples: no few-shot example from their source sample
    consult_prompt_cfg: PromptConfig | None = None   # prompts of the consult adapter (AgentConfig.consult)


def initial_messages(rc: RunContext, s: Sample, pcfg: PromptConfig | None = None) -> tuple[list[dict], list[str]]:
    final, hist = texts_for(s, rc.transcripts, prefer=rc.text_source)
    fs = ""
    if rc.pool is not None and rc.prompt_cfg.fewshot_k > 0:
        allowed = usable_apps(s.installed_apps, rc.catalog) if rc.prompt_cfg.fewshot_installed_only else None
        ex = {s.id} | rc.exclude_fewshot
        if rc.exclude_origin and s.origin:
            ex.add(s.origin)
        fs = rc.pool.block_for(final, rc.prompt_cfg.fewshot_k, exclude=ex, allowed_apps=allowed)
    tools = rc.catalog.allowed_tools(s.installed_tool_names())
    return build_messages(s, rc.catalog, final, hist, fs, pcfg or rc.prompt_cfg, tools), tools


def spoken_clock(final: str | None, hist: list[str] | None) -> list[str]:
    # times from the last request; if it has none, from the latest earlier user turn that has one
    for t in [final] + list(reversed(hist or [])):
        c = clock_times(t)
        if c:
            return c
    return []


def user_text(final: str | None, hist: list[str] | None) -> str:
    return " | ".join(t for t in list(hist or []) + [final or ""] if t)


def build_episode(rc: RunContext, s: Sample, uid: str | None = None, seed: int = 0, env=None) -> Episode:
    if env is None:
        env = make_env(rc.sp, s.env_id)     # exactly one ToolEnv per sample (synthetic: its source's data)
    msgs, tools = initial_messages(rc, s)
    final, hist = texts_for(s, rc.transcripts, prefer=rc.text_source)
    guard = Guard(rc.catalog, tools, s.context, rc.guard_cfg, rc.formats, clock=spoken_clock(final, hist),
                  text=user_text(final, hist))
    ep = Episode(uid or s.id, s, env, msgs, guard, rc.prompt_cfg, seed)
    if rc.consult_prompt_cfg is not None:
        ep.consult_init = initial_messages(rc, s, rc.consult_prompt_cfg)[0]
        ep.consult_thoughts = rc.consult_prompt_cfg.thoughts
    return ep


def run_split(rc: RunContext, backend: Backend, samples: list[Sample], out: str | Path | None = None,
              trace: str | Path | None = None, batch_size: int = 16, keep_messages: bool = False,
              done_ids: set[str] | None = None, log_every: int = 50) -> list[dict]:
    agent = Agent(backend, rc.agent_cfg)
    results: list[dict] = []
    todo = [s for s in samples if not done_ids or s.id not in done_ids]
    t_start = time.time()
    tf = open(trace, "a", encoding="utf-8") if trace else None
    of = open(out, "a", encoding="utf-8") if out else None
    try:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            eps = [build_episode(rc, s) for s in chunk]
            agent.run(eps)
            for e in eps:
                r = e.result()
                results.append(r)
                if of:
                    of.write(json.dumps({"id": r["id"], "tool_calls": r["tool_calls"],
                                         "state_hash": r["state_hash"]}, ensure_ascii=False) + "\n")
                    of.flush()
                if tf:
                    t = dict(r)
                    if keep_messages:
                        t["messages"] = e.messages
                    tf.write(json.dumps(t, ensure_ascii=False) + "\n")
                    tf.flush()
            n = len(results)
            if log_every and (n % log_every < batch_size or n == len(todo)):
                el = time.time() - t_start
                print(f"[run] {n}/{len(todo)} samples, {el:.0f}s, {el / max(1, n):.2f}s/sample", flush=True)
    finally:
        if tf:
            tf.close()
        if of:
            of.close()
    return results


def finalize_submission(path: str | Path, sp: SplitPaths, samples: list[Sample]) -> int:
    from .data import read_jsonl
    rows = {}
    if Path(path).exists():
        for r in read_jsonl(path):
            rows.setdefault(r["id"], r)
    out, missing = [], 0
    for s in samples:
        r = rows.get(s.id)
        if r is None:
            missing += 1            # e.g. after a crash: empty calls + initial state hash
            env = make_env(sp, s.id)
            out.append({"id": s.id, "tool_calls": [], "state_hash": env.get_state_hash()})
        else:
            out.append({"id": s.id, "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]})
    write_jsonl(path, out)
    return missing
