"""Inference behind predict.sh and predict_notebook.ipynb: load both models once, then run samples one at a time.

`time` per row = ms from creating the sample's ToolEnv to get_state_hash(), the span BTC's notebook times.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .data import Sample, SplitPaths, make_env, oracle_texts, read_jsonl, resolve_split

PREFERRED_SPLITS = ("private_test", "public_test")
ADAPTERS = "adapters"         # <llm>/adapters/{main,consult}: the base model in <llm> with two adapters (consult)


@dataclass
class PredictConfig:
    data: str = "/data"
    split: str | None = None
    out: str = "/result/submission.jsonl"
    backend: str = "vllm"                     # vllm | hf | oracle
    llm: str = "/code/saved_models/llm"
    lora: str | None = None                   # default: <llm>/adapters/main if there is one
    consult_lora: str | None = None           # default: <llm>/adapters/consult if there is one (agent.consult)
    asr: str = "auto"                         # auto | whisper | gipformer | oracle
    asr_model: str = "/code/saved_models/asr"
    asr_lora: str | None = None
    asr_device: str = "cuda"
    asr_beams: int = 0                        # 0 = lifemate_asr.json next to the ASR weights, else 1
    asr_prompt: str = "auto"                  # auto (same file) | on | off: history context for the final request
    asr2_model: str = "/code/saved_models/asr2"   # second ASR (gipformer) for fusion of the final request
    fuse: str = "auto"                        # auto: on when assets/fuse.json and asr2_model exist | on | off
    assets: str = "/code/assets"
    config: str = "/code/configs/docker.json"         # the submission's settings (configs/default.json: training)
    bias: str | None = "/code/assets/bias.json"
    gpu_mem: float = 0.6
    max_model_len: int = 16384
    seed: int = 42
    warmup: bool = True
    limit: int = 0
    trace: str | None = None

    def override(self, argv: list[str] | None) -> "PredictConfig":
        ap = argparse.ArgumentParser(prog="predict")
        for f in fields(self):
            v = getattr(self, f.name)
            flag = "--" + f.name.replace("_", "-")
            if isinstance(v, bool):
                ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
            else:
                ap.add_argument(flag, type=str if v is None else type(v), default=v)
        return PredictConfig(**vars(ap.parse_args(argv or [])))


def resolve_adapters(cfg: PredictConfig) -> PredictConfig:
    """An LLM folder with adapters/main and adapters/consult holds the base model and two adapters: use them unless
    --lora / --consult-lora name others. A merged model (no such folder) is used as it is."""
    root = Path(cfg.llm) / ADAPTERS

    def found(name):
        return str(root / name) if (root / name / "adapter_config.json").is_file() else None
    return replace(cfg, lora=cfg.lora or found("main"), consult_lora=cfg.consult_lora or found("consult"))


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_split(data: str = "/data", split: str | None = None) -> SplitPaths:
    root = Path(data)
    if split is None and root.is_dir():
        # the guide mounts the parent: /data/private_test/private_test.jsonl
        subs = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / f"{d.name}.jsonl").exists())
        if subs:
            split = next((s for s in PREFERRED_SPLITS if s in subs), subs[0] if len(subs) == 1 else None)
            if split is None:
                raise SystemExit(f"several splits under {root}: {subs}, pass --split")
    return resolve_split(root, split)


def _read_json(path) -> dict | None:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_warmup_wav(path: Path, seconds: float = 1.0, sr: int = 16000):
    import struct
    import wave
    rng = random.Random(0)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{n}h", *(rng.randint(-300, 300) for _ in range(n))))


class Predictor:
    def __init__(self, cfg: PredictConfig):
        from .agent import Agent, AgentConfig
        from .fewshot import FewShotPool
        from .guard import GuardConfig
        from .prompt import PromptConfig
        from .timeutil import FormatBook

        t0 = time.perf_counter()
        cfg = resolve_adapters(cfg)
        self.cfg = cfg
        conf = _read_json(cfg.config) or {}
        self.pcfg = PromptConfig.from_json(conf.get("prompt"))
        from .prompt import apply_model_prompt
        src = apply_model_prompt(self.pcfg, cfg.lora, cfg.llm)  # thoughts on iff the model was trained with them
        print(f"[predict] thoughts={self.pcfg.thoughts} ({src or 'config'})", flush=True)
        self.gcfg = GuardConfig.from_json(conf.get("guard"))
        self.acfg = AgentConfig.from_json(conf.get("agent"))
        self.cpcfg = None                     # prompts of the consult: same settings, its thought format
        clora = cfg.consult_lora if cfg.backend == "vllm" else None
        if clora or self.acfg.consult_self:
            from .prompt import consult_prompt
            self.cpcfg, src = consult_prompt(self.pcfg, clora, self.acfg.consult_thoughts)
            print(f"[predict] consult={self.acfg.consult or 'off'} by {clora or 'the main model again'} "
                  f"thoughts={self.cpcfg.thoughts} ({src or 'config'})", flush=True)
        b = _read_json(cfg.bias)
        if b:
            self.acfg.use_type_scoring = bool(b.get("use_type_scoring", True))
            self.acfg.bias_first, self.acfg.bias_later = b.get("bias_first", {}), b.get("bias_later", {})
        assets = Path(cfg.assets)
        self.pool = FewShotPool.load(assets / "fewshot_pool.jsonl") if (assets / "fewshot_pool.jsonl").exists() else None
        f = _read_json(assets / "formats.json")
        self.formats = FormatBook.from_json(f) if f else None

        # vLLM first, so its engine process starts before this process touches CUDA
        self.llm = self._load_llm()
        self.asr = self._load_asr()
        self.agent = Agent(self.llm, self.acfg) if self.llm is not None else None
        if cfg.warmup:
            self._warmup()
        self.sp: SplitPaths | None = None
        self.rc = None
        self.last = None
        self.load_s = time.perf_counter() - t0

    def _load_llm(self):
        c = self.cfg
        if c.backend == "oracle":
            return None                     # needs the split's labels, built in open_split
        from .llm import make_backend
        if c.backend == "vllm":
            return make_backend("vllm", model=c.llm, lora=c.lora, max_model_len=c.max_model_len,
                                gpu_memory_utilization=c.gpu_mem, seed=c.seed, consult_lora=c.consult_lora,
                                consult_self=self.acfg.consult_self)
        return make_backend(c.backend, model=c.llm, lora=c.lora, consult_self=self.acfg.consult_self)

    def _load_asr(self):
        c = self.cfg
        self.asr_prompt = False
        self.asr2, self.fuse_params, self.fuser = None, None, None
        if c.asr == "oracle":
            return None
        self._load_fusion()
        from .asr.context import read_settings
        from .asr.engines import make_engine
        st = read_settings(c.asr_lora, c.asr_model)
        beams = int(c.asr_beams or st.get("beams") or 1)
        self.asr_prompt = c.asr_prompt == "on" or (c.asr_prompt == "auto" and bool(st.get("prompt_final")))
        print(f"[predict] asr beams={beams} prompt_final={self.asr_prompt}", flush=True)
        return make_engine(c.asr, c.asr_model, c.asr_lora, c.asr_device, beams)

    def _load_fusion(self):
        c = self.cfg
        if c.fuse == "off" or c.backend == "oracle":
            return
        from .asr.fuse import read_params
        params = read_params(c.assets)
        if params is None or not float(params.get("delta", float("inf"))) < float("inf"):
            if c.fuse == "on":
                print("[predict] fuse=on but no usable assets/fuse.json: fusion off", flush=True)
            return
        if not Path(c.asr2_model).exists():
            print(f"[predict] no second ASR at {c.asr2_model}: fusion off", flush=True)
            return
        from .asr.engines import make_engine
        try:
            self.asr2 = make_engine("auto", c.asr2_model, None, "cpu", 1)
        except Exception as ex:                # e.g. an empty saved_models/asr2: run without fusion
            print(f"[predict] second ASR not loaded ({type(ex).__name__}: {ex}): fusion off", flush=True)
            return
        self.fuse_params = params
        print(f"[predict] fusion on: delta={params['delta']} beta={params.get('beta', 0.0)}", flush=True)

    def _fuse(self, s: Sample, final: str) -> str:
        if self.asr2 is None or self.llm is None or not s.audio:
            return final
        try:
            from .asr.fuse import Fuser
            if self.fuser is None:
                p = self.fuse_params
                self.fuser = Fuser(self.llm, float(p["delta"]), float(p.get("beta", 0.0)))
            second = self.asr2.transcribe([str(self.sp.audio(s.audio))], batch_size=1)[0]
            return self.fuser.fuse_many([(s, final, second)])[0]
        except Exception as ex:                # fusion is an extra: never lose the transcript over it
            print(f"[fuse] {s.id}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            return final

    def _warmup(self):
        # first calls compile kernels / allocate buffers; keep that out of the per-sample time
        from .llm import TYPE_CANDIDATES, GenRequest, ScoreRequest
        if self.asr is not None:
            p = Path(tempfile.gettempdir()) / "lifemate_warmup.wav"
            _write_warmup_wav(p)
            self.asr.transcribe([str(p)], batch_size=1)
            if self.asr_prompt:                # the prompted path too, so a version problem shows here, not per sample
                self.asr.transcribe([str(p)], batch_size=1, prompts=["Dạ, em nhắn Zalo cho Lan rồi ạ."])
            if self.asr2 is not None:
                self.asr2.transcribe([str(p)], batch_size=1)
        if self.llm is not None and self.llm.needs_prompt:
            # a couple thousand tokens, so long prefill has run once too
            prompt = self.llm.render([{"role": "user", "content": "Xin chào. " * 400}])
            self.llm.generate([GenRequest(None, prompt, max_new_tokens=8)])
            if self.llm.has_adapter("consult"):    # vLLM loads an adapter on first use: not inside a sample's time
                self.llm.generate([GenRequest(None, prompt, max_new_tokens=8, adapter="consult")])
            if self.acfg.use_type_scoring:
                self.llm.score([ScoreRequest(None, prompt, list(TYPE_CANDIDATES))])

    def open_split(self, sp: SplitPaths):
        from .agent import Agent
        from .catalog import Catalog
        from .data import Transcripts, load_samples, load_toolenv_class
        from .llm import OracleBackend
        from .runner import RunContext
        self.sp = sp
        cat = Catalog.load(sp.catalog)
        cls = load_toolenv_class(sp)        # import the split's own zaic_tools now, not inside the timer
        self.rc = RunContext(sp, cat, Transcripts(), "asr", self.pool, self.formats, self.pcfg, self.gcfg, self.acfg,
                             consult_prompt_cfg=self.cpcfg)
        if self.cfg.backend == "oracle":
            self.llm = OracleBackend({s.id: s for s in load_samples(sp)}, cat)
            self.agent = Agent(self.llm, self.acfg)
        return cls

    def transcribe(self, s: Sample) -> tuple[str, list[str]]:
        if self.asr is None:
            return oracle_texts(s)
        from .asr.transcribe import utterance_prompts, utterances
        paths = [str(self.sp.audio(rel)) if rel else "" for _, rel in utterances(s)]
        prompts = utterance_prompts(s, self.asr_prompt)
        todo = [p for p in paths if p]
        tprompts = [q for p, q in zip(paths, prompts) if p]
        kw = {"prompts": tprompts} if any(tprompts) else {}
        try:
            got = self.asr.transcribe(todo, batch_size=max(1, len(todo)), **kw) if todo else []
        except Exception as ex:
            print(f"[asr] {s.id}: batch failed ({ex}), retrying one by one", file=sys.stderr, flush=True)
            got = []
            for p in todo:
                try:
                    got += self.asr.transcribe([p], batch_size=1)
                except Exception as ex2:
                    print(f"[asr] {s.id}: {p}: {ex2}", file=sys.stderr, flush=True)
                    got.append("")
        it = iter(got)
        texts = [next(it) if p else "" for p in paths]
        if self.asr2 is not None:
            texts[0] = self._fuse(s, texts[0])
        return texts[0], texts[1:]

    def run_one(self, item: dict | Sample) -> tuple[list, str]:
        from .runner import build_episode
        if self.rc is None:
            raise RuntimeError("call open_split() first")
        s = item if isinstance(item, Sample) else Sample(item)
        started = time.time()
        env = make_env(self.sp, s.id)       # the only ToolEnv for this sample
        ep = None
        try:
            final, hist = self.transcribe(s)
            self.rc.transcripts.rows[s.id] = {"id": s.id, "final": final, "history": hist}
            ep = build_episode(self.rc, s, env=env)
            ep.t0 = started                 # the agent deadline counts ASR time too
            self.agent.run([ep])
        except Exception as ex:             # report whatever already ran on env
            print(f"[predict] {s.id}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            try:                            # an ask held back by defer_ask still belongs to the answer
                if ep is not None and ep.deferred_ask is not None:
                    self.agent._stop(ep, "error")
            except Exception as ex2:
                print(f"[predict] {s.id}: ask not sent: {ex2}", file=sys.stderr, flush=True)
        self.last = ep
        return env.get_tool_calls(), env.get_state_hash()

    def predict(self, item: dict) -> dict:
        t1 = time.perf_counter()
        try:
            calls, h = self.run_one(item)
        except Exception as ex:             # ToolEnv itself failed: nothing valid to report
            print(f"[predict] {item.get('id')}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            calls, h = [], ""
        ms = int((time.perf_counter() - t1) * 1000)
        return {"id": item["id"], "tool_calls": calls, "state_hash": h, "time": ms}

    def trace_row(self, item: dict, row: dict) -> dict:
        t = self.rc.transcripts.rows.get(item["id"], {}) if self.rc else {}
        ep = self.last if self.last is not None and self.last.sample.id == item["id"] else None
        return {"id": item["id"], "time": row["time"], "asr_final": t.get("final"), "asr_history": t.get("history"),
                "stop_reason": ep.stop_reason if ep else "error", "steps": ep.step if ep else 0,
                "events": ep.events if ep else [], "tool_calls": row["tool_calls"]}


def run(cfg: PredictConfig) -> list[dict]:
    from .compat import hide_old_torchao
    hide_old_torchao()
    seed_everything(cfg.seed)
    t0 = time.perf_counter()
    sp = find_split(cfg.data, cfg.split)
    cases = read_jsonl(sp.jsonl)
    if cfg.limit:
        cases = cases[:cfg.limit]
    print(f"[predict] split={sp.split} samples={len(cases)} dir={sp.dir}", flush=True)
    print(f"[predict] config {json.dumps(asdict(cfg), ensure_ascii=False)}", flush=True)
    pr = Predictor(cfg)
    pr.open_split(sp)
    print(f"[time] model loading: {time.perf_counter() - t0:.1f}s", flush=True)

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr = open(cfg.trace, "w", encoding="utf-8") if cfg.trace else None
    rows: list[dict] = []
    t1 = time.perf_counter()
    try:
        with open(out, "w", encoding="utf-8") as f:
            for i, item in enumerate(cases, 1):
                r = pr.predict(item)
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if tr:
                    tr.write(json.dumps(pr.trace_row(item, r), ensure_ascii=False) + "\n")
                    tr.flush()
                if i % 50 == 0 or i == len(cases):
                    print(f"[predict] {i}/{len(cases)}, {time.perf_counter() - t1:.0f}s", flush=True)
    finally:
        if tr:
            tr.close()
    ms = sorted(r["time"] for r in rows) or [0]
    print(f"[time] prediction: {time.perf_counter() - t1:.1f}s for {len(rows)} samples, "
          f"mean {sum(ms) / len(ms):.0f} ms, p95 {ms[int(0.95 * (len(ms) - 1))]} ms, max {ms[-1]} ms", flush=True)
    slow = sum(1 for x in ms if x > 30000)
    if slow:
        print(f"[warn] {slow} samples took over 30 s", flush=True)
    print(f"[predict] {len(rows)} rows -> {out}", flush=True)
    return rows


def main(argv=None):
    run(PredictConfig().override(argv))


if __name__ == "__main__":
    main()
