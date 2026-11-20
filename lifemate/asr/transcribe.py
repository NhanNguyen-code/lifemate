"""Transcribe every user turn (final request + history) into a resumable jsonl cache."""
from __future__ import annotations

import time
from pathlib import Path

from ..data import Sample, SplitPaths, load_samples, read_ids, read_jsonl, resolve_split


def utterances(s: Sample) -> list[tuple[str, str]]:
    out = [("final", s.audio)]
    for k, t in enumerate(s.user_turns()):
        out.append((f"h{k}", t.get("audio", "")))
    return out


def utterance_prompts(s: Sample, prompt_final: bool = False) -> list[str | None]:
    """Whisper prompt per utterance, in utterances() order: only the final request gets one (history turns cost
    little when misheard, and one prompt per call would slow them down)."""
    out: list[str | None] = [None] * (1 + len(s.user_turns()))
    if prompt_final:
        from .context import prompt_for
        out[0] = prompt_for(s) or None
    return out


def transcribe_samples(engine, sp: SplitPaths, samples: list[Sample], out: str | Path,
                       batch_size: int = 16, oracle: bool = False, log_every: int = 200,
                       prompt_final: bool = False, history_from: str | Path | None = None) -> int:
    """history_from: take the history turns from this transcript file and transcribe only the final request."""
    import json
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [s for s in samples if s.id not in done]
    old = {r["id"]: r for r in read_jsonl(history_from)} if history_from else None
    t0, n_utt = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            items = []
            for s in chunk:
                prompts = utterance_prompts(s, prompt_final)
                for (key, rel), p in zip(utterances(s), prompts):
                    if old is not None and key != "final":
                        continue
                    items.append((s, key, rel, p))
            if oracle:
                texts = []
                for s, key, _, _ in items:
                    fin, h = s.transcripts()
                    texts.append((fin if key == "final" else h[int(key[1:])]) or "")
            else:
                paths = [str(sp.audio(rel)) for _, _, rel, _ in items]
                prompts = [p for _, _, _, p in items]
                texts = engine.transcribe(paths, batch_size=max(batch_size, 1),
                                          **({"prompts": prompts} if any(prompts) else {}))
            n_utt += len(items)
            by = {}
            for s in chunk:
                hist = list((old.get(s.id) or {}).get("history") or []) if old is not None else []
                by[s.id] = {"id": s.id, "final": "", "history": hist}
            for (s, key, _, _), t in zip(items, texts):
                if key == "final":
                    by[s.id]["final"] = t
                else:
                    by[s.id]["history"].append(t)
            for s in chunk:
                f.write(json.dumps(by[s.id], ensure_ascii=False) + "\n")
            f.flush()
            done_n = min(i + batch_size, len(todo))
            if log_every and (done_n % log_every < batch_size or done_n == len(todo)):
                el = time.time() - t0
                print(f"[asr] {done_n}/{len(todo)} samples, {n_utt} utterances, {el:.0f}s", flush=True)
    return len(todo)


def main(a):
    sp = resolve_split(a.data, a.split)
    samples = load_samples(sp, read_ids(a.ids_file))
    if a.limit:
        samples = samples[: a.limit]
    engine = None
    if a.asr != "oracle":
        from .engines import make_engine
        if not a.model:
            raise SystemExit("--model is required unless --asr oracle")
        engine = make_engine(a.asr, a.model, a.lora, a.device, a.num_beams)
    n = transcribe_samples(engine, sp, samples, a.out, a.batch_size, oracle=a.asr == "oracle",
                           prompt_final=getattr(a, "prompt_final", False),
                           history_from=getattr(a, "history_from", None))
    print(f"[asr] {n} new samples -> {a.out}")
