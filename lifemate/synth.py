"""Synthetic train samples from train states (development only; the rules allow get_state on train).

For each train sample a teacher LLM sees what the agent would see (context, installed tools, earlier turns), the apps'
data (ToolEnv.get_state(), train only), the real request and its gold calls, and writes K new requests of the same
kind with other details. The teacher then solves each new request as the agent, through the guard, on fresh train
ToolEnvs: run A with the usual few-shot examples (the source sample is usually the nearest), run B with none. A request
is kept only if both runs make the same calls and reach the same state, with no tool error, no guard rejection and a
thought at every step. Its label is run A's calls plus the results of replaying them on one more fresh ToolEnv.

--check-ids first runs the same two-run filter on real samples with gold (val): how many pass and how close the kept
ones are to gold. Below --min-kept-score the command stops with exit code 3 before generating anything.
Output: samples in train.jsonl form (+ env_id, origin) and their thoughts ({id, step, thought}); reruns resume.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from .agent import AgentConfig
from .catalog import Catalog
from .data import Sample, load_samples, make_env, oracle_texts, read_ids, read_jsonl, resolve_split
from .fewshot import FewShotPool, render_trace
from .guard import GuardConfig
from .prompt import PromptConfig, compact_json, render_context, render_history
from .rationale import clean
from .runner import RunContext, build_episode
from .timeutil import FormatBook

REQ_INSTRUCTION = (
    "Bạn tạo dữ liệu luyện tập cho một trợ lý điều khiển ứng dụng điện thoại bằng giọng nói tiếng Việt. Cho trước: "
    "bối cảnh người dùng, các ứng dụng đã cài cùng tool của chúng, dữ liệu hiện có trong các ứng dụng, hội thoại "
    "trước (nếu có), một yêu cầu thật ở lượt cuối và các bước trợ lý đã làm đúng cho yêu cầu đó.\n"
    "Hãy viết __K__ yêu cầu MỚI mà chính người dùng này có thể nói ở lượt cuối:\n"
    "- Cùng kiểu việc với yêu cầu thật nhưng đổi chi tiết: người khác, món khác, số tiền, thời gian, điều kiện hoặc "
    "cách nói khác. Được có một yêu cầu làm việc khác, miễn là dùng các ứng dụng đã cài.\n"
    "- Chỉ dùng tên, mục, số liệu có thật trong dữ liệu ở trên; việc phải làm được bằng các tool đã cài.\n"
    "- Nếu yêu cầu thật thiếu thông tin nên trợ lý phải hỏi lại, các yêu cầu mới cũng thiếu đúng loại thông tin đó. "
    "Nếu yêu cầu thật không ứng dụng nào làm được, các yêu cầu mới cũng là việc không làm được.\n"
    "- Viết như lời nói tự nhiên, ngắn gọn (1–2 câu), có thể có điều kiện \"nếu... thì...\" hoặc nhắc tới hội thoại "
    "trước.\n"
    "Chỉ trả về JSON: {\"requests\": [\"...\", \"...\"]}")


def state_text(state: dict, apps: list[str], per_app: int = 900, total: int = 6000) -> str:
    parts, used = [], 0
    for a in apps:
        if a.startswith("_") or a not in state:
            continue
        t = f"{a}: {compact_json(state[a], per_app, 8)}"
        if used + len(t) > total:
            break
        parts.append(t)
        used += len(t)
    return "\n".join(parts) or "(trống)"


def request_messages(s: Sample, cat: Catalog, state: dict, k: int, pcfg: PromptConfig) -> list[dict]:
    final, hist = oracle_texts(s)
    apps = list(s.installed_apps.keys())
    parts = [render_context(s.context, pcfg),
             "# Ứng dụng đã cài\n" + cat.render_tools(cat.allowed_tools(s.installed_tool_names()), apps),
             "# Dữ liệu hiện có trong các ứng dụng\n" + state_text(state, apps),
             render_history(s, hist, pcfg) or "# Hội thoại trước\n(không có)",
             f"# Yêu cầu thật ở lượt cuối\n{final}\n# Các bước trợ lý đã làm đúng\n"
             + render_trace(s.gold_calls or [], cat),
             f"Viết {k} yêu cầu mới."]
    return [{"role": "system", "content": REQ_INSTRUCTION.replace("__K__", str(k))},
            {"role": "user", "content": "\n\n".join(parts)}]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def parse_requests(text: str, k: int, original: str = "") -> list[str]:
    t = (text or "").strip()
    reqs = None
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(t[i:j + 1])
            reqs = obj.get("requests") if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            reqs = None
    if not isinstance(reqs, list):           # a plain list of lines
        reqs = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"')
                for ln in t.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", ln)]
    out, seen = [], {_norm(original)}
    for r in reqs:
        if not isinstance(r, str):
            continue
        r = " ".join(r.split())
        if not 8 <= len(r) <= 400 or _norm(r) in seen:
            continue
        seen.add(_norm(r))
        out.append(r)
    return out[:k]


def synth_sample(src: Sample, j: int, text: str) -> Sample:
    r = copy.deepcopy(src.raw)
    r.update(id=f"{src.id}~s{j}", env_id=src.env_id, origin=src.id, audio="", audio_text=text)
    r.pop("label", None)
    return Sample(r)


def same_calls(a: list[dict], b: list[dict], fuzzy_map: dict) -> bool:
    """Same tools in the same order and the same arguments; free-text (x-match fuzzy) ones as the scorer compares them."""
    from .scorer import call_match
    nfc = lambda calls: json.loads(unicodedata.normalize("NFC", json.dumps(calls, ensure_ascii=False)))  # noqa: E731
    a, b = nfc(a), nfc(b)
    return len(a) == len(b) and all(x["name"] == y["name"] and call_match(x, y, fuzzy_map) for x, y in zip(a, b))


def verdict(a: dict, b: dict | None, empty_ok: bool, fuzzy_map: dict | None = None, need_b: bool = True) -> str:
    """'ok', or why a run (pair) is dropped; run A alone is judged with need_b=False."""
    if a["stop_reason"] not in ("finish", "ask"):
        return f"stop_{a['stop_reason']}"
    if any(ev["kind"] == "reject" for ev in a["events"]):
        return "rejected"
    if any(ev["kind"] == "call" and not ev.get("ok") for ev in a["events"]):
        return "tool_error"
    if any(ev["kind"] == "call" and ev.get("noop") for ev in a["events"]):
        return "noop"                        # a call that changed nothing (unknown ID, short balance...): gold has none
    if not a["tool_calls"] and not empty_ok:
        return "no_call"
    if any(not ev.get("thought") for ev in a["events"] if ev["kind"] in ("call", "finish")):
        return "no_thought"
    if b is None:
        return "ok" if not need_b else "no_run_b"
    if b["stop_reason"] != a["stop_reason"] or a["state_hash"] != b["state_hash"] \
            or not same_calls(a["tool_calls"], b["tool_calls"], fuzzy_map or {}):
        return "disagree"
    return "ok"


def run_all(rc: RunContext, backend, acfg: AgentConfig, samples: list[Sample], batch: int) -> dict[str, dict]:
    from .agent import Agent
    agent, out = Agent(backend, acfg), {}
    for i in range(0, len(samples), batch):
        eps = [build_episode(rc, s) for s in samples[i:i + batch]]
        agent.run(eps)
        for e in eps:                        # keep the small result, drop the env
            out[e.sample.id] = e.result()
    return out


def thoughts_of(res: dict) -> list[str]:
    th = [clean(ev["thought"]) for ev in res["events"] if ev["kind"] == "call"]
    if res["stop_reason"] == "finish":
        th += [clean(ev["thought"]) for ev in res["events"] if ev["kind"] == "finish"][-1:]
    return th


def replay(sp, env_id: str, calls: list[dict]) -> tuple[list, str]:
    env = make_env(sp, env_id)
    results = [env.call(c["name"], c["arguments"]) for c in calls]
    return results, env.get_state_hash()


def check(a, sp, cat, backend, acfg, rc_a, rc_b) -> dict:
    """The two-run filter on real samples with gold: keep rate and quality of the kept ones."""
    from .scorer import Scorer
    samples = [s for s in load_samples(sp, read_ids(a.check_ids)) if s.gold_calls is not None]
    if a.limit:
        samples = samples[:a.limit]
    t0 = time.time()
    ra = run_all(rc_a, backend, acfg, samples, a.batch)
    fmap = cat.fuzzy_map()
    pre = [s for s in samples if verdict(ra[s.id], None, not s.gold_calls, fmap, need_b=False) == "ok"]
    rb = run_all(rc_b, backend, acfg, pre, a.batch)
    sc = Scorer(sp, cat, cache=a.hash_cache)
    score = {s.id: sc.score_one(s, ra[s.id]["tool_calls"], ra[s.id]["state_hash"])["score"] for s in samples}
    sc.save_cache()
    why = Counter(verdict(ra[s.id], rb.get(s.id), not s.gold_calls, fmap) for s in samples)
    kept = [s.id for s in pre if verdict(ra[s.id], rb[s.id], not s.gold_calls, fmap) == "ok"]
    n = len(samples)
    rep = {"n": n, "score_run_a": round(100 * sum(score.values()) / max(1, n), 2),
           "kept": len(kept), "keep_rate": round(100 * len(kept) / max(1, n), 1),
           "kept_score": round(100 * sum(score[i] for i in kept) / max(1, len(kept)), 2),
           "kept_perfect": round(100 * sum(score[i] == 1 for i in kept) / max(1, len(kept)), 1),
           "why": dict(why), "min_kept_score": a.min_kept_score, "seconds": round(time.time() - t0)}
    rep["passed"] = rep["kept_score"] >= a.min_kept_score
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate synth")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="teacher model folder (scripts/download_models.py teacher)")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default=None, help="synthetic samples (train.jsonl form)")
    ap.add_argument("--thoughts-out", default=None)
    ap.add_argument("--k", type=int, default=5, help="new requests per train sample")
    ap.add_argument("--ids-file", default=None, help="source samples (default: all of train)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=300, help="source samples per round; output written after each")
    ap.add_argument("--batch", type=int, default=40,
                    help="episodes per agent batch; keep their prompts inside the KV cache so steps reuse them")
    ap.add_argument("--check-ids", default=None, help="real samples with gold (val) to measure the filter on first")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--min-kept-score", type=float, default=85.0)
    ap.add_argument("--report", default=None, help="check result (json); a passed one is not run again")
    ap.add_argument("--hash-cache", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args(argv)

    sp = resolve_split(a.data, "train")
    cat = Catalog.load(sp.catalog)
    samples = [s for s in load_samples(sp) if s.gold_calls is not None]
    by_id = {s.id: s for s in samples}
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config and Path(a.config).exists() else {}
    pcfg = PromptConfig.from_json(cfg.get("prompt"))
    pcfg.thoughts = True
    gcfg = GuardConfig.from_json(cfg.get("guard"))
    acfg = AgentConfig.from_json(cfg.get("agent"))
    acfg.deadline_s, acfg.temperature, acfg.use_type_scoring = 0.0, 0.0, False
    acfg.max_new_tokens = max(acfg.max_new_tokens, 512)
    fb, ask = FormatBook(), cat.ask_format()
    for s in samples:
        fb.observe(s.gold_calls or [], ask)
    pool = FewShotPool.from_samples(samples, cat)
    pcfg_b = PromptConfig.from_json(pcfg.to_json())
    pcfg_b.fewshot_k = 0
    rc_a = RunContext(sp, cat, None, "oracle", pool, fb, pcfg, gcfg, acfg)
    rc_b = RunContext(sp, cat, None, "oracle", pool, fb, pcfg_b, gcfg, acfg)
    fmap = cat.fuzzy_map()

    from .llm import GenRequest, VLLMBackend
    backend = None

    def teacher():
        nonlocal backend
        if backend is None:
            backend = VLLMBackend(model=a.model, chat_kwargs={"enable_thinking": False},
                                  max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                                  max_num_seqs=a.max_num_seqs)
        return backend

    rp = Path(a.report) if a.report else None
    prev = json.loads(rp.read_text(encoding="utf-8")) if rp and rp.exists() else None
    if prev is not None and "kept_score" in prev:   # measured before: judge it again with today's threshold
        prev["min_kept_score"] = a.min_kept_score
        prev["passed"] = prev["kept_score"] >= a.min_kept_score
        print(f"[synth] earlier check: kept runs score {prev['kept_score']}, passed={prev['passed']}", flush=True)
        if a.check_ids and not prev["passed"]:
            sys.exit(3)
    if a.check_ids and prev is None:
        rep = check(a, sp, cat, teacher(), acfg, rc_a, rc_b)
        print("[synth] check on real samples: " + json.dumps(rep, ensure_ascii=False), flush=True)
        if rp:
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        if not rep["passed"]:
            print(f"[synth] kept runs score {rep['kept_score']} < {a.min_kept_score}: not generating", flush=True)
            sys.exit(3)
    if a.check_only:
        return
    if not a.out or not a.thoughts_out:
        ap.error("--out and --thoughts-out are needed to generate")

    out, th_out = Path(a.out), Path(a.thoughts_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog = Path(str(out) + ".progress")
    done = set(prog.read_text(encoding="utf-8").split()) if prog.exists() else set()
    src = samples
    if a.ids_file:
        keep = set(read_ids(a.ids_file))
        src = [s for s in src if s.id in keep]
    if a.limit:
        src = src[:a.limit]
    todo = [s for s in src if s.id not in done]
    have = {r["id"] for r in read_jsonl(out)} if out.exists() else set()   # a round cut off after writing
    print(f"[synth] {len(todo)} source samples to do ({len(done)} done), {a.k} new requests each", flush=True)
    stats, t0, shown = Counter(), time.time(), 0
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]
        be = teacher()
        reqs, metas = [], []
        for s in chunk:
            state = make_env(sp, s.env_id).get_state()       # train only (rules: get_state is for train)
            prompt = be.render(request_messages(s, cat, state, a.k, pcfg))
            if len(be.tok.encode(prompt, add_special_tokens=False)) > a.max_model_len - 700:
                stats["prompt_too_long"] += 1
                continue
            reqs.append(GenRequest(None, prompt, "", 0.0, 600))
            metas.append(s)
        syn = []
        for s, o in zip(metas, be.generate(reqs) if reqs else []):
            texts = parse_requests(o, a.k, oracle_texts(s)[0] or "")
            stats["requests"] += len(texts)
            syn += [synth_sample(s, j, t) for j, t in enumerate(texts)]
        ra = run_all(rc_a, be, acfg, syn, a.batch)
        empty_ok = {x.id: not by_id[x.origin].gold_calls for x in syn}
        pre = [x for x in syn if verdict(ra[x.id], None, empty_ok[x.id], fmap, need_b=False) == "ok"]
        rb = run_all(rc_b, be, acfg, pre, a.batch)
        rows, trows = [], []
        for x in syn:
            v = verdict(ra[x.id], rb.get(x.id), empty_ok[x.id], fmap)
            if v == "ok" and x.id in have:
                v = "already_written"
            if v == "ok":
                calls = ra[x.id]["tool_calls"]
                results, h = replay(sp, x.env_id, calls)
                if h != ra[x.id]["state_hash"] or not all(isinstance(r, dict) and r.get("ok")
                                                          and r.get("done") is not False for r in results):
                    v = "replay_mismatch"
                else:
                    r = dict(x.raw)
                    r["label"] = {"tool_calls": calls, "tool_results": results}
                    rows.append(r)
                    trows += [{"id": x.id, "step": i, "thought": t} for i, t in enumerate(thoughts_of(ra[x.id]))]
                    if shown < 5:
                        shown += 1
                        print(f"  {x.origin} -> {x.audio_text}\n     " + " → ".join(c["name"] for c in calls),
                              flush=True)
            stats[v] += 1
        with open(out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(th_out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in trows)
        with open(prog, "a", encoding="utf-8") as f:
            f.writelines(s.id + "\n" for s in chunk)
        print(f"[synth] {c0 + len(chunk)}/{len(todo)} sources, {time.time() - t0:.0f}s, {dict(stats)}", flush=True)
    n = sum(1 for _ in read_jsonl(out)) if out.exists() else 0
    print(f"[synth] done: {n} synthetic samples in {out}", flush=True)


if __name__ == "__main__":
    main()
