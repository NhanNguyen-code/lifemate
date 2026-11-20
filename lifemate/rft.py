"""Rejection-sampling fine-tuning data: K rollouts per train sample, keep the ones scoring 100."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate rft")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default=None, help="use train_ids only (skip val)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--p-asr", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=0, help="0: pick from GPU memory")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fewshot-k", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--prioritize", default=None, help="score report of the greedy run: sample failed ids first")
    ap.add_argument("--max-keep-per-sample", type=int, default=2)
    ap.add_argument("--hard-only", action=argparse.BooleanOptionalAction, default=True,
                    help="skip samples whose k rollouts were all perfect without any guard fix: nothing to learn")
    ap.add_argument("--keep-rejected", action=argparse.BooleanOptionalAction, default=True,
                    help="keep perfect rollouts that hit guard rejections, minus the rejected attempts, so the "
                         "model learns the guard's corrections")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from .actions import finish
    from .agent import Agent, AgentConfig, Episode
    from .catalog import Catalog
    from .cli import _load_cfg
    from .data import Transcripts, load_samples, make_env, resolve_split, texts_for, write_jsonl
    from .fewshot import FewShotPool
    from .guard import Guard, GuardConfig
    from .llm import make_backend
    from .prompt import PromptConfig, assistant_message, build_messages, strip_rejected
    from .runner import spoken_clock, user_text
    from .scorer import Scorer
    from .timeutil import FormatBook

    sp = resolve_split(a.data, "train")
    cat = Catalog.load(sp.catalog)
    samples = load_samples(sp)
    val = set()
    if a.split_file:
        val = set(json.loads(Path(a.split_file).read_text(encoding="utf-8")).get("val_ids") or [])
    train = [s for s in samples if s.id not in val and s.gold_calls is not None]
    if a.prioritize:
        rep = json.loads(Path(a.prioritize).read_text(encoding="utf-8"))
        bad = {p["id"] for p in rep["per_sample"] if p["score"] < 1}
        train.sort(key=lambda s: (s.id not in bad, s.id))
    if a.max_samples:
        train = train[: a.max_samples]
    cfg = _load_cfg(a.config)
    pcfg = PromptConfig.from_json(cfg.get("prompt"))
    from .prompt import apply_model_prompt
    f = apply_model_prompt(pcfg, a.lora, a.model)
    print(f"[rft] thoughts={pcfg.thoughts} ({f or 'config'})", flush=True)
    if a.fewshot_k is not None:
        pcfg.fewshot_k = a.fewshot_k
    gcfg = GuardConfig.from_json(cfg.get("guard"))
    acfg = AgentConfig.from_json(cfg.get("agent"))
    acfg.temperature = a.temperature
    acfg.deadline_s = 1e9
    pool = FewShotPool.from_samples(train, cat)
    fb = FormatBook()
    for s in train:
        fb.observe(s.gold_calls or [], cat.ask_format())
    tr = Transcripts.load(a.transcripts) if a.transcripts else None
    backend = make_backend("vllm", model=a.model, lora=a.lora, gpu_memory_utilization=a.gpu_mem)
    agent = Agent(backend, acfg)
    scorer = Scorer(sp, cat)

    def episode(s, j):
        rng = random.Random(f"rft:{s.id}:{j}")
        use_asr = tr is not None and tr.has(s.id) and rng.random() < a.p_asr
        final, hist = texts_for(s, tr if use_asr else None, prefer="asr" if use_asr else "oracle")
        fs = pool.block_for(final, pcfg.fewshot_k, exclude={s.id} | val) if pcfg.fewshot_k else ""
        tools = cat.allowed_tools(s.installed_tool_names())
        msgs = build_messages(s, cat, final, hist, fs, pcfg, tools)
        guard = Guard(cat, tools, s.context, gcfg, fb, clock=spoken_clock(final, hist), text=user_text(final, hist))
        return Episode(f"{s.id}#{j}", s, make_env(sp, s.id), msgs, guard, pcfg, seed=j)

    kept, stats = [], {"samples": 0, "episodes": 0, "perfect": 0, "perfect_after_reject": 0, "kept": 0,
                       "solved_samples": 0}
    eps_all = [(s, j) for s in train for j in range(a.k)]
    by_sample: dict[str, list] = {}
    n_perfect: dict[str, int] = {}
    from .compat import auto_batch
    a.batch_size = a.batch_size or auto_batch(a.k)      # the k rollouts of a sample share its prompt prefix
    for i in range(0, len(eps_all), a.batch_size):
        chunk = [episode(s, j) for s, j in eps_all[i:i + a.batch_size]]
        agent.run(chunk)
        for e in chunk:
            stats["episodes"] += 1
            rejected = any(ev["kind"] == "reject" for ev in e.events)
            if e.stop_reason not in ("finish", "ask") or (rejected and not a.keep_rejected):
                continue
            r = scorer.score_one(e.sample, e.env.get_tool_calls(), e.env.get_state_hash())
            if r["score"] < 1.0:
                continue
            stats["perfect"] += 1
            stats["perfect_after_reject"] += rejected
            n_perfect[e.sample.id] = n_perfect.get(e.sample.id, 0) + (not rejected)
            msgs = strip_rejected(e.messages, e.n_messages_init)
            if e.stop_reason == "finish":
                msgs.append(assistant_message(finish()))
            sig = tuple(m["content"] for m in msgs[e.n_messages_init:] if m["role"] == "assistant")
            lst = by_sample.setdefault(e.sample.id, [])
            if sig in {x[0] for x in lst}:
                continue
            lst.append((sig, msgs, rejected))
        print(f"[rft] {min(i + a.batch_size, len(eps_all))}/{len(eps_all)} rollouts, perfect {stats['perfect']}",
              flush=True)
    stats["easy_samples"] = 0
    for sid, lst in by_sample.items():
        if a.hard_only and n_perfect.get(sid, 0) >= a.k:
            stats["easy_samples"] += 1          # solved every time with no guard fix
            continue
        lst.sort(key=lambda x: not x[2])        # guard-corrected rollouts first: they teach the most
        for sig, msgs, rej in lst[: a.max_keep_per_sample]:
            kept.append({"messages": msgs, "meta": {"id": sid, "aug": ["rft", "rft_fixed"] if rej else ["rft"]}})
    stats["samples"] = len(train)
    stats["kept"] = len(kept)
    stats["solved_samples"] = len(by_sample)
    write_jsonl(a.out, kept)
    scorer.save_cache()
    print(json.dumps(stats, ensure_ascii=False), f"-> {a.out}")


if __name__ == "__main__":
    main()
