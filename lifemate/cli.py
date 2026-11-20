"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data import Transcripts, read_ids, read_jsonl, resolve_split, write_jsonl

ROOT = Path(__file__).resolve().parent.parent


def _load_cfg(path: str | None) -> dict:
    if not path:
        p = ROOT / "configs" / "default.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ctx(a, split: str | None = None):
    from .catalog import Catalog
    from .data import load_samples
    sp = resolve_split(a.data, split or a.split)
    cat = Catalog.load(sp.catalog)
    ids = read_ids(getattr(a, "ids_file", None))
    samples = load_samples(sp, ids)
    if getattr(a, "limit", None):
        samples = samples[: a.limit]
    return sp, cat, samples


def _assets(a, exclude: set[str] | None = None):
    from .catalog import Catalog
    from .data import load_samples
    from .fewshot import FewShotPool
    from .timeutil import FormatBook
    exclude = exclude or set()
    d = Path(a.assets) if getattr(a, "assets", None) else None
    pool = fb = None
    if d and (d / "fewshot_pool.jsonl").exists() and not exclude:
        pool = FewShotPool.load(d / "fewshot_pool.jsonl")
    if d and (d / "formats.json").exists() and not exclude:
        fb = FormatBook.from_json(json.loads((d / "formats.json").read_text(encoding="utf-8")))
    if pool is None or fb is None:
        try:
            tsp = resolve_split(a.data, "train")
        except FileNotFoundError:
            tsp = None
        if tsp is not None:
            tcat = Catalog.load(tsp.catalog)
            ts = [s for s in load_samples(tsp) if s.id not in exclude]
            if pool is None:
                pool = FewShotPool.from_samples(ts, tcat)
            if fb is None:
                fb = FormatBook()
                for s in ts:
                    fb.observe(s.gold_calls or [], tcat.ask_format())
        elif exclude:
            raise SystemExit("--exclude-fewshot-ids needs the train split under --data")
    return pool, fb


def cmd_experiments(a):
    from .catalog import Catalog
    from .experiments import print_report, run_all
    sp, cat, samples = _ctx(a)
    others = {}
    for name in ("public_test", "private_test"):
        try:
            others[name] = Catalog.load(resolve_split(a.data, name).catalog)
        except FileNotFoundError:
            pass
    rep = run_all(sp, cat, samples, others, a.out, a.max_tests)
    text = print_report(rep)
    txt = Path(a.out).with_suffix(".txt")
    txt.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nreport -> {txt} (full json: {a.out})")


def cmd_make_split(a):
    from .splits import leave_apps_out, random_split
    sp, cat, samples = _ctx(a)
    if a.holdout == "apps":
        sp_ = leave_apps_out(samples, cat, a.n_apps, a.seed, a.oos_frac)
    else:
        sp_ = random_split(samples, a.n_val, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(sp_, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.out).with_suffix(".val_ids.txt").write_text("\n".join(sp_["val_ids"]) + "\n", encoding="utf-8")
    print(f"{sp_['type']}: train={len(sp_['train_ids'])} val={len(sp_['val_ids'])} "
          f"held_apps={sp_['held_apps']} -> {a.out}")


def cmd_build_assets(a):
    from .data import load_samples
    from .catalog import Catalog
    from .fewshot import FewShotPool
    from .timeutil import FormatBook
    sp = resolve_split(a.data, "train")
    cat = Catalog.load(sp.catalog)
    samples = load_samples(sp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    FewShotPool.from_samples(samples, cat).save(out / "fewshot_pool.jsonl")
    fb = FormatBook()
    for s in samples:
        fb.observe(s.gold_calls or [], cat.ask_format())
    (out / "formats.json").write_text(json.dumps(fb.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"assets -> {out} ({len(samples)} train samples)")


def _run_ctx(a, sp, cat):
    from .agent import AgentConfig
    from .guard import GuardConfig
    from .prompt import PromptConfig
    from .runner import RunContext
    cfg = _load_cfg(a.config)
    pcfg = PromptConfig.from_json(cfg.get("prompt"))
    if getattr(a, "fewshot_installed", False):
        pcfg.fewshot_installed_only = True
    if getattr(a, "backend", "vllm") in ("vllm", "hf"):
        from .prompt import apply_model_prompt
        f = apply_model_prompt(pcfg, getattr(a, "lora", None), getattr(a, "model", None))
        print(f"[infer] thoughts={pcfg.thoughts} ({f or 'config'})", flush=True)
    gcfg = GuardConfig.from_json(cfg.get("guard"))
    acfg = AgentConfig.from_json(cfg.get("agent"))
    if a.fewshot_k is not None:
        pcfg.fewshot_k = a.fewshot_k
    if getattr(a, "bias", None):
        b = json.loads(Path(a.bias).read_text(encoding="utf-8"))
        acfg.use_type_scoring = bool(b.get("use_type_scoring", True))
        acfg.bias_first, acfg.bias_later = b.get("bias_first", {}), b.get("bias_later", {})
    if getattr(a, "temperature", None) is not None:
        acfg.temperature = a.temperature
    excl = set(read_ids(a.exclude_fewshot_ids) or [])
    if not excl and sp.split == "train" and getattr(a, "ids_file", None):
        # dev run on train: keep the evaluated samples out of few-shot / formats, otherwise scores are inflated
        excl = set(read_ids(a.ids_file) or [])
        print(f"[infer] excluding {len(excl)} eval samples from few-shot pool and formats")
    pool, fb = _assets(a, excl)
    tr = Transcripts.load(a.transcripts) if a.transcripts else None
    if getattr(a, "consult", None):
        acfg.consult = "" if a.consult == "off" else a.consult
    if getattr(a, "consult_self", False):
        acfg.consult_self = True
    if getattr(a, "consult_thoughts", None):
        acfg.consult_thoughts = a.consult_thoughts
    cpcfg = None
    clora = getattr(a, "consult_lora", None)
    if clora or acfg.consult_self:              # same prompt settings, the thought format of the consult
        from .prompt import consult_prompt
        cpcfg, f = consult_prompt(pcfg, clora, acfg.consult_thoughts)
        print(f"[infer] consult={acfg.consult or 'off'} thoughts={cpcfg.thoughts} "
              f"({f or ('config' if clora else 'the main model again')})", flush=True)
    return RunContext(sp, cat, tr, a.text, pool, fb, pcfg, gcfg, acfg, excl, consult_prompt_cfg=cpcfg)


def _backend(a, samples, cat, consult_self: bool = False):
    from .llm import OracleBackend, ScriptedBackend, make_backend
    if a.backend == "empty":
        return ScriptedBackend({})
    if a.backend == "oracle":
        return OracleBackend({s.id: s for s in samples}, cat)
    if a.backend == "vllm":
        return make_backend("vllm", model=a.model, lora=a.lora, max_model_len=a.max_model_len,
                            gpu_memory_utilization=a.gpu_mem, seed=a.seed, dtype=a.dtype,
                            consult_lora=getattr(a, "consult_lora", None), consult_self=consult_self)
    return make_backend("hf", model=a.model, lora=a.lora, load_4bit=a.load_4bit,
                        consult_lora=getattr(a, "consult_lora", None), consult_self=consult_self)


def cmd_infer(a):
    from .runner import finalize_submission, run_split
    sp, cat, samples = _ctx(a)
    rc = _run_ctx(a, sp, cat)
    backend = _backend(a, samples, cat, consult_self=rc.agent_cfg.consult_self)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.resume and out.exists():
        done = {r["id"] for r in read_jsonl(out)}
    elif out.exists():
        out.unlink()
    if a.trace and Path(a.trace).exists() and not a.resume:
        Path(a.trace).unlink()
    from .compat import auto_batch
    bs = a.batch_size or auto_batch()
    print(f"[infer] batch size {bs}", flush=True)
    run_split(rc, backend, samples, out, a.trace, bs, a.keep_messages, done)
    if not a.ids_file and not a.limit:
        miss = finalize_submission(out, sp, samples)
        if miss:
            print(f"[warn] {miss} samples missing from the run, written as empty")
    print(f"submission -> {out}")


def cmd_score(a):
    from .scorer import Scorer, format_report
    sp, cat, samples = _ctx(a)
    rows = read_jsonl(a.submission)
    rep = Scorer(sp, cat, a.hash_cache).score(rows, samples, a.only_submitted)
    print(format_report(rep))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {a.report}")


def cmd_trace_stats(a):
    from .catalog import Catalog
    from .tracestats import format_stats, load, stats
    cat = Catalog.load(resolve_split(a.data, a.split).catalog) if a.data else None
    for t in a.trace:
        st = stats(load(t), cat)
        print(f"== {Path(t).name}")
        print(format_stats(st), flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_inspect(a):
    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    bad = [p for p in rep["per_sample"] if p["score"] < 1]
    if a.tag:
        bad = [p for p in bad if any(t.startswith(a.tag) for t in p["tags"])]
    bad.sort(key=lambda p: p["score"])
    trace = {}
    if a.trace:
        trace = {r["id"]: r for r in read_jsonl(a.trace)}
    for p in bad[: a.show]:
        print(f"{p['id']}  score={p['score']:.2f} hash_ok={p['hash_ok']} {p['category']}  {p['tags']}")
        t = trace.get(p["id"])
        if t:
            for c in t["tool_calls"]:
                print(f"     pred: {c['name']} {json.dumps(c.get('arguments'), ensure_ascii=False)}")
            for ev in t.get("events", []):
                if ev.get("thought"):
                    print(f"     think [{ev.get('name', ev['kind'])}]: {ev['thought']}")
        if a.data:
            from .data import load_samples
            sp = resolve_split(a.data, a.split)
            s = next(iter(load_samples(sp, [p["id"]])), None)
            if s:
                print(f"     text: {s.audio_text}")
                for c in s.gold_calls or []:
                    print(f"     gold: {c['name']} {json.dumps(c.get('arguments'), ensure_ascii=False)}")


def cmd_build_sft(a):
    from .catalog import Catalog
    from .data import Sample, load_samples
    from .fewshot import FewShotPool
    from .prompt import PromptConfig
    from .sft_data import SFTConfig, build_rows
    sp = resolve_split(a.data, "train")
    cat = Catalog.load(sp.catalog)
    samples = load_samples(sp)
    cfg = _load_cfg(a.config)
    pcfg = PromptConfig.from_json(cfg.get("prompt"))
    if a.fewshot_k is not None:
        pcfg.fewshot_k = a.fewshot_k
    scfg = SFTConfig.from_json(cfg.get("sft"))
    for k in ("p_asr", "n_aug", "p_rename", "p_tool_rename", "p_distractor", "p_drop_ask", "p_fewshot_other_app",
              "seed"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(scfg, k, v)
    held, val = set(), set()
    if a.split_file:
        spl = json.loads(Path(a.split_file).read_text(encoding="utf-8"))
        held, val = set(spl.get("held_apps") or []), set(spl.get("val_ids") or [])
    train = [s for s in samples if s.id not in val]
    pool = FewShotPool.from_samples(train, cat)           # real samples only
    if a.extra_samples:
        extra = list({r["id"]: Sample(r) for path in a.extra_samples for r in read_jsonl(path)}.values())
        ok = [x for x in extra if x.origin not in val
              and not any(c["name"].split(".")[0] in held for c in (x.gold_calls or []))]
        print(f"[build-sft] {len(ok)}/{len(extra)} synthetic samples added")
        train += ok
    tr = Transcripts.load(a.transcripts) if a.transcripts else None
    th = None
    if a.thoughts:
        from .sft_data import load_thoughts
        th = load_thoughts(a.thoughts)
        print(f"[build-sft] thoughts for {len(th)} samples")
    pcfg.thoughts = th is not None          # the prompt asks for thoughts only if the rows have them
    rows, stat = build_rows(train, cat, tr, pool, pcfg, scfg, held, val, thoughts=th)
    write_jsonl(a.out, rows)
    print(f"sft rows: {stat} -> {a.out}")


def cmd_transcribe(a):
    from .asr.transcribe import main as tmain
    tmain(a)


def cmd_asr_eval(a):
    from .asr.metrics import evaluate
    sp, cat, samples = _ctx(a)
    tr = Transcripts.load(a.transcripts)
    rep = evaluate(samples, tr, show=a.show_numbers)
    wrong = rep.pop("number_mismatches")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    for line in wrong:
        print(line)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_fuse(a):
    """Final request = primary ASR + secondary ASR merged with LLM scores (lifemate/asr/fuse.py)."""
    from .asr import fuse as F
    sp, cat, samples = _ctx(a)
    prim, sec = Transcripts.load(a.primary), Transcripts.load(a.secondary)
    scores = {}
    if a.scores and Path(a.scores).exists():
        scores = json.loads(Path(a.scores).read_text(encoding="utf-8"))
    # only samples whose two transcripts disagree need the LLM, and only once (cache)
    todo = [s for s in samples if s.id not in scores and prim.has(s.id) and sec.has(s.id)
            and F.regions(prim.final(s) or "", sec.final(s) or "")[1]]
    if todo:
        from .llm import make_backend
        if a.backend == "vllm":
            be = make_backend("vllm", model=a.model, lora=a.lora, max_model_len=a.max_model_len,
                              gpu_memory_utilization=a.gpu_mem, seed=0)
        else:
            be = make_backend("hf", model=a.model, lora=a.lora)
        scores.update(F.score_samples(F.Fuser(be), todo, prim, sec))
        if a.scores:
            Path(a.scores).parent.mkdir(parents=True, exist_ok=True)
            Path(a.scores).write_text(json.dumps(scores), encoding="utf-8")
    print(f"[fuse] {len(samples)} samples, {len(scores)} with spans to score")
    if a.tune:
        params, table = F.tune(samples, prim, sec, scores)
        base = F.final_wer(samples, {s.id: prim.final(s) or "" for s in samples if prim.has(s.id)})
        print(f"[fuse] WER of the final request: primary {base:.2f}, secondary "
              f"{F.final_wer(samples, {s.id: sec.final(s) or '' for s in samples if sec.has(s.id)}):.2f}")
        for d, b, w in table[:8]:
            print(f"   delta={d:<5} beta={b:<5} -> {w:.2f}")
        params["wer_primary"] = round(base, 3)
        if a.params_out:
            Path(a.params_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.params_out).write_text(json.dumps(params), encoding="utf-8")
        print("[fuse] chosen:", params)
    elif a.params:
        params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    else:
        params = {"delta": a.delta, "beta": a.beta}
    finals = F.fused_finals(samples, prim, sec, scores, float(params["delta"]), float(params.get("beta", 0.0)))
    rows = [{"id": s.id, "final": finals[s.id], "history": prim.history(s) or []} for s in samples if s.id in finals]
    write_jsonl(a.out, rows)
    changed = sum(1 for s in samples if s.id in finals and finals[s.id] != (prim.final(s) or ""))
    print(f"[fuse] {changed}/{len(rows)} final requests changed -> {a.out}")


def cmd_asr_settings(a):
    from .asr.context import read_settings, write_settings
    kw = {}
    if a.beams is not None:
        kw["beams"] = a.beams
    if a.prompt_final is not None:
        kw["prompt_final"] = a.prompt_final == "on"
    cur = write_settings(a.dir, **kw) if kw else read_settings(a.dir)
    print(json.dumps(cur, ensure_ascii=False))


def cmd_validate(a):
    import tempfile
    from .data import read_jsonl, write_jsonl
    sp = resolve_split(a.data, a.split)
    sub = Path(a.submission).resolve()
    rows = read_jsonl(sub)
    bad_time = []
    if any("time" in r for r in rows):
        # `time` comes from the docker guide; check it here and give the split's validator the original fields
        bad_time = [r.get("id") for r in rows if not isinstance(r.get("time"), int) or r["time"] < 0]
        print(f"time field: {len(rows) - len(bad_time)}/{len(rows)} ok" + (f", bad: {bad_time[:5]}" if bad_time else ""))
        tmp = Path(tempfile.mkdtemp(prefix="validate_")) / sub.name
        write_jsonl(tmp, ({k: r[k] for k in ("id", "tool_calls", "state_hash") if k in r} for r in rows))
        sub = tmp
    cmd = [sys.executable, "-m", "zaic_tools", "--split", sp.split, "--root", "..", "--validate", str(sub)]
    print(" ".join(cmd), f"(cwd={sp.dir})")
    code = subprocess.call(cmd, cwd=sp.dir)
    sys.exit(code or (1 if bad_time else 0))


PASSTHROUGH = {
    "train": ("lifemate.train_lora", "LoRA SFT"),
    "merge-lora": ("lifemate.merge_lora", "merge a LoRA adapter into the base LLM"),
    "soup-lora": ("lifemate.soup_lora", "weighted sum of two LoRA adapters as one adapter (no training)"),
    "rft": ("lifemate.rft", "rejection-sampling fine-tuning data"),
    "calibrate": ("lifemate.calibrate", "grid-search action biases on val"),
    "count-params": ("lifemate.count_params", "check LLM < 5B and total < 6B"),
    "finetune-whisper": ("lifemate.asr.finetune_whisper", "LoRA fine-tune PhoWhisper on train audio"),
    "merge-whisper": ("lifemate.asr.merge_whisper", "merge a PhoWhisper LoRA (for Docker)"),
    "asr-manifest": ("lifemate.asr.export_manifest", "kaldi-style data dirs for gipformer/icefall"),
    "predict": ("lifemate.predict", "docker inference path: /data/<split> -> /result/submission.jsonl"),
    "compare": ("lifemate.compare", "compare two submissions and their time fields"),
    "diag": ("lifemate.diag", "counts behind the guard's optional rules (CPU)"),
    "diff-runs": ("lifemate.diffruns", "per-sample score changes between two scored runs"),
    "effect": ("lifemate.effect", "score change of a run only on the samples its options acted on (noise apart)"),
    "synth": ("lifemate.synth", "synthetic train samples on train states, solved twice by a teacher LLM"),
    "rationale": ("lifemate.rationale", "a teacher LLM writes a short thought for every gold step of train"),
}


def _common(p, split_default="train"):
    p.add_argument("--data", required=True, help="dir holding the split dirs, or a split dir itself")
    p.add_argument("--split", default=split_default)
    p.add_argument("--ids-file", default=None)
    p.add_argument("--limit", type=int, default=None)


def _utf8_stdio():                      # Windows pipes default to the ANSI code page
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None):
    _utf8_stdio()
    from .compat import hide_old_torchao
    hide_old_torchao()
    ap = argparse.ArgumentParser(prog="python -m lifemate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("experiments", help="dataset stats + day-1 checks (CPU)")
    _common(p)
    p.add_argument("--out", default="outputs/experiments.json")
    p.add_argument("--max-tests", type=int, default=60)
    p.set_defaults(fn=cmd_experiments)

    p = sub.add_parser("make-split", help="leave-apps-out or random dev split")
    _common(p)
    p.add_argument("--holdout", choices=["apps", "random"], default="apps")
    p.add_argument("--n-apps", type=int, default=10)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--oos-frac", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/split_apps.json")
    p.set_defaults(fn=cmd_make_split)

    p = sub.add_parser("build-assets", help="few-shot pool + value formats from train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="assets")
    p.set_defaults(fn=cmd_build_assets)

    p = sub.add_parser("transcribe", help="transcribe every user turn (final + history)")
    _common(p)
    p.add_argument("--asr", choices=["auto", "gipformer", "whisper", "oracle"], default="auto",
                   help="auto: onnx files -> gipformer, whisper config -> PhoWhisper")
    p.add_argument("--model", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--prompt-final", action="store_true",
                   help="Whisper prompt for the final request: assistant reply + names from the history")
    p.add_argument("--history-from", default=None,
                   help="copy history turns from this transcript file, transcribe only the final request")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("fuse", help="final request from two ASR outputs, chosen span by span with LLM scores")
    _common(p)
    p.add_argument("--primary", required=True, help="transcripts of the main ASR (PhoWhisper)")
    p.add_argument("--secondary", required=True, help="transcripts of the second ASR (gipformer)")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--scores", default=None, help="json cache of the LLM scores (reused, so tuning reruns are free)")
    p.add_argument("--tune", action="store_true", help="choose delta/beta by WER of the final request (labelled split)")
    p.add_argument("--params-out", default=None, help="write the chosen delta/beta here (assets/fuse.json)")
    p.add_argument("--params", default=None, help="fuse.json to apply")
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fuse)

    p = sub.add_parser("asr-settings", help="decoding settings stored next to ASR weights (lifemate_asr.json)")
    p.add_argument("--dir", required=True, help="LoRA adapter or merged model folder")
    p.add_argument("--beams", type=int, default=None)
    p.add_argument("--prompt-final", choices=["on", "off"], default=None,
                   help="history context as Whisper prompt for the final request")
    p.set_defaults(fn=cmd_asr_settings)

    p = sub.add_parser("asr-eval", help="WER / no-tone WER / number accuracy vs gold transcripts")
    _common(p)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--show-numbers", type=int, default=0, help="print this many utterances whose numbers differ")
    p.add_argument("--report", default=None, help="also write the numbers to this json file")
    p.set_defaults(fn=cmd_asr_eval)

    p = sub.add_parser("infer", help="run the agent on a split -> submission.jsonl")
    p.add_argument("--fewshot-installed", action="store_true",
                   help="few-shot examples only with apps installed for the sample")
    _common(p)
    p.add_argument("--backend", choices=["vllm", "hf", "oracle", "empty"], default="vllm",
                   help="empty: no calls (format check); oracle: replay gold (train only)")
    p.add_argument("--model", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--consult-lora", default=None,
                   help="second adapter on the same base, asked at the first stop without a write (agent.consult)")
    p.add_argument("--consult", choices=["write", "handover", "off"], default=None, help="overrides agent.consult")
    p.add_argument("--consult-self", action="store_true",
                   help="no second adapter: the main model is asked again in the other prompt format")
    p.add_argument("--consult-thoughts", choices=["on", "off"], default=None, help="prompt format of the consult")
    p.add_argument("--text", choices=["asr", "oracle"], default="asr")
    p.add_argument("--transcripts", default=None)
    p.add_argument("--assets", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--bias", default=None, help="json written by `calibrate`")
    p.add_argument("--fewshot-k", type=int, default=None)
    p.add_argument("--exclude-fewshot-ids", default=None, help="ids to keep out of the few-shot pool")
    p.add_argument("--batch-size", type=int, default=0, help="0: pick from GPU memory")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--trace", default=None)
    p.add_argument("--keep-messages", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("score", help="score a submission on a labelled split")
    _common(p)
    p.add_argument("--submission", required=True)
    p.add_argument("--only-submitted", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--hash-cache", default="outputs/gold_hash_cache.json")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("trace-stats", help="what a run did, from its trace (no labels): give-ups, retries, votes")
    p.add_argument("--trace", nargs="+", required=True)
    p.add_argument("--data", default=None, help="with --split: the catalog tells reads from writes")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=None, help="json of the last trace's numbers")
    p.set_defaults(fn=cmd_trace_stats)

    p = sub.add_parser("inspect", help="show the worst samples of a score report")
    p.add_argument("--report", required=True)
    p.add_argument("--trace", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--tag", default=None, help="error tag prefix, e.g. write_arg, extra_read")
    p.add_argument("--show", type=int, default=20)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("build-sft", help="build SFT conversations (+ augmentation)")
    p.add_argument("--data", required=True)
    p.add_argument("--split-file", default=None, help="make-split output: drops val ids and held-out apps")
    p.add_argument("--transcripts", nargs="+", default=None,
                   help="ASR transcripts of train; later files win per id (e.g. untuned gipformer, then dev PhoWhisper on val)")
    p.add_argument("--config", default=None)
    p.add_argument("--fewshot-k", type=int, default=None)
    p.add_argument("--p-asr", dest="p_asr", type=float, default=None)
    p.add_argument("--n-aug", dest="n_aug", type=int, default=None)
    p.add_argument("--p-rename", dest="p_rename", type=float, default=None)
    p.add_argument("--p-tool-rename", dest="p_tool_rename", type=float, default=None)
    p.add_argument("--p-distractor", dest="p_distractor", type=float, default=None)
    p.add_argument("--p-fewshot-other-app", dest="p_fewshot_other_app", type=float, default=None,
                   help="share of rows whose few-shot examples avoid the sample's own apps")
    p.add_argument("--p-drop-ask", dest="p_drop_ask", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--thoughts", nargs="+", default=None,
                   help="`rationale` / `synth` output: a thought before every gold step")
    p.add_argument("--extra-samples", nargs="+", default=None,
                   help="`synth` output: synthetic samples added to train (with --split-file: only those whose "
                        "source is not in val and whose calls use no held-out app)")
    p.add_argument("--out", default="outputs/sft_train.jsonl")
    p.set_defaults(fn=cmd_build_sft)

    p = sub.add_parser("validate", help="validate a submission with the split's zaic_tools")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="public_test")
    p.add_argument("--submission", required=True)
    p.set_defaults(fn=cmd_validate)

    for name, (mod, help_) in PASSTHROUGH.items():
        sub.add_parser(name, help=help_, add_help=False)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASSTHROUGH:
        import importlib
        importlib.import_module(PASSTHROUGH[argv[0]][0]).main(argv[1:])
        return
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
