"""A teacher LLM writes a short thought for every gold step of train (synthetic SFT data, development only).

The teacher sees what the agent sees at that step (context, history, request, earlier steps with their results) plus
the correct next step, and explains it in one short sentence: which detail leads to it, with the numbers of any sum
or comparison. One jsonl row per step {id, step, thought}; written per batch of whole samples, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .actions import action_text, call_to_action, finish
from .catalog import Catalog
from .data import load_samples, oracle_texts, read_ids, read_jsonl, resolve_split
from .prompt import PromptConfig, compact_json, render_context, render_history

INSTRUCTION = (
    "Bạn viết lý do ngắn cho từng bước của một trợ lý điều khiển ứng dụng trên điện thoại. Cho trước hội thoại, "
    "các bước trợ lý đã làm kèm kết quả, và BƯỚC TIẾP THEO ĐÚNG. Hãy viết suy nghĩ của trợ lý ngay trước khi làm "
    "bước đó: một câu tiếng Việt, tối đa 30 từ, nêu chi tiết nào trong lời người dùng hoặc kết quả trước dẫn tới "
    "bước này. Bước hiển nhiên thì chỉ cần vài từ. Nếu có tính toán hay so sánh thì ghi rõ con số (ví dụ: "
    "150000 + 420000 = 570000 lớn hơn số dư 500000 nên chuyển từ tài khoản tiết kiệm). Không nêu tên tool hay tên "
    "tham số, không nói là đã biết đáp án, không chép lại hành động. Chỉ trả về câu lý do.")


def step_list(s, cat: Catalog) -> list[dict]:
    """The agent's steps on a sample: the gold actions, then finish unless they end with an ask."""
    acts = [call_to_action(c, cat) for c in (s.gold_calls or [])]
    if not acts or acts[-1]["type"] != "ask":
        acts.append(finish())
    return acts


def teacher_messages(s, pcfg: PromptConfig, k: int, acts: list[dict]) -> list[dict]:
    final, hist = oracle_texts(s)
    parts = [render_context(s.context, pcfg)]
    h = render_history(s, hist, pcfg)
    if h:
        parts.append(h)
    parts.append(f"Yêu cầu hiện tại của người dùng: {final}")
    results = s.gold_results
    if k:
        done = []
        for i in range(k):
            r = results[i] if i < len(results) else {"ok": True}
            done.append(f"{i + 1}. {action_text(acts[i])}\n   Kết quả: {compact_json(r, 1200, 12)}")
        parts.append("Các bước đã làm:\n" + "\n".join(done))
    else:
        parts.append("Chưa làm bước nào.")
    nxt = acts[k]
    parts.append("BƯỚC TIẾP THEO ĐÚNG: " + ("kết thúc, không gọi thêm tool nào." if nxt["type"] == "finish"
                                             else action_text(nxt)))
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": "\n\n".join(parts)}]


def clean(text: str, max_chars: int = 240) -> str | None:
    t = " ".join((text or "").split())
    t = re.sub(r"^(lý do|suy nghĩ|thought)\s*[:：]\s*", "", t, flags=re.I).strip('"“”\' ')
    if len(t) > max_chars:
        cut = t[:max_chars]
        end = cut.rfind(".")
        t = cut[: end + 1] if end >= 40 else cut.rstrip() + "…"
    return t or None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate rationale")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", required=True, help="teacher model folder (scripts/download_models.py teacher)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=2000, help="steps per vLLM call; whole samples, written after each")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args(argv)

    sp = resolve_split(a.data, a.split)
    cat = Catalog.load(sp.catalog)
    samples = [s for s in load_samples(sp, read_ids(a.ids_file) if a.ids_file else None) if s.gold_calls is not None]
    if a.limit:
        samples = samples[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [s for s in samples if s.id not in done]
    print(f"[rationale] {len(todo)} samples to do, {len(done)} already in {out}", flush=True)
    if not todo:
        return

    from .llm import GenRequest, VLLMBackend
    pcfg = PromptConfig()
    backend = VLLMBackend(model=a.model, chat_kwargs={"enable_thinking": False},
                          max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                          max_num_seqs=a.max_num_seqs)
    budget = a.max_model_len - a.max_new_tokens - 8
    t0, n_steps, n_long, shown = time.time(), 0, 0, 0
    batch: list[tuple[str, int, str | None]] = []

    def flush():
        nonlocal n_steps, shown
        reqs = [GenRequest(None, p, "", 0.0, a.max_new_tokens) for _, _, p in batch if p is not None]
        outs = iter(backend.generate(reqs))
        rows = []
        for sid, k, p in batch:
            rows.append({"id": sid, "step": k, "thought": clean(next(outs)) if p is not None else None})
        with open(out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in rows[:max(0, 6 - shown)]:
            print(f"  {r['id']} #{r['step']}: {r['thought']}", flush=True)
        shown += min(len(rows), max(0, 6 - shown))
        n_steps += len(rows)
        el = time.time() - t0
        print(f"[rationale] {n_steps} steps, {el:.0f}s", flush=True)
        batch.clear()

    for s in todo:
        acts = step_list(s, cat)
        for k in range(len(acts)):
            prompt = backend.render(teacher_messages(s, pcfg, k, acts))
            if len(backend.tok.encode(prompt, add_special_tokens=False)) > budget:
                prompt, n_long = None, n_long + 1
            batch.append((s.id, k, prompt))
        if len(batch) >= a.chunk:
            flush()
    if batch:
        flush()
    print(f"[rationale] done: {n_steps} steps ({n_long} prompts too long, left without a thought) -> {out}")


if __name__ == "__main__":
    main()
