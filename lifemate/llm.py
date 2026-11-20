"""LLM backends: vLLM (offline), HF transformers, and oracle / scripted replay for tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import TYPE_PREFIX, TYPES, action_text, call_to_action, finish

STOP = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"]


@dataclass
class GenRequest:
    episode: Any
    prompt: str
    prefix: str = ""
    temperature: float = 0.0
    max_new_tokens: int = 384
    seed: int = 0
    adapter: str | None = None          # "consult": the second adapter (AgentConfig.consult); None: the main model


@dataclass
class ScoreRequest:
    episode: Any
    prompt: str
    candidates: list[str] = field(default_factory=list)


class Backend:
    needs_prompt = True

    def render(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def generate(self, reqs: list[GenRequest]) -> list[str]:
        raise NotImplementedError

    def score(self, reqs: list[ScoreRequest]) -> list[list[float]]:
        raise NotImplementedError

    def has_adapter(self, name: str) -> bool:
        return False


def _strip_prefix(full: str, prefix: str) -> str:
    return full[len(prefix):] if prefix and full.startswith(prefix) else full


class OracleBackend(Backend):
    needs_prompt = False

    def __init__(self, samples: dict, catalog):
        self.samples = samples
        self.cat = catalog

    def _plan(self, ep) -> list[dict]:
        s = self.samples[ep.sample.id]
        acts = [call_to_action(c, self.cat) for c in (s.gold_calls or [])]
        if not acts or acts[-1]["type"] != "ask":
            acts.append(finish())
        return acts

    def render(self, messages):
        return ""

    def generate(self, reqs):
        out = []
        for r in reqs:
            plan = self._plan(r.episode)
            a = plan[min(r.episode.step, len(plan) - 1)]
            out.append(_strip_prefix(action_text(a), r.prefix))
        return out

    def score(self, reqs):
        res = []
        for r in reqs:
            plan = self._plan(r.episode)
            t = plan[min(r.episode.step, len(plan) - 1)]["type"]
            res.append([0.0 if c.rstrip('"') == t else -20.0 for c in r.candidates])
        return res


class ScriptedBackend(Backend):
    """Fixed answers per sample, in order. consult: the answers of the second adapter, kept apart."""
    needs_prompt = False

    def __init__(self, script: dict[str, list[str]], consult: dict[str, list[str]] | None = None):
        self.script = script
        self.consult = consult
        self.calls: dict[str, int] = {}
        self.consult_calls: dict[str, int] = {}

    def render(self, messages):
        return ""

    def has_adapter(self, name):
        return name == "consult" and self.consult is not None

    def generate(self, reqs):
        out = []
        for r in reqs:
            sid = r.episode.sample.id
            book, seqs = ((self.consult_calls, self.consult or {}) if r.adapter == "consult"
                          else (self.calls, self.script))
            k = book.get(sid, 0)
            book[sid] = k + 1
            seq = seqs.get(sid, [])
            text = seq[k] if k < len(seq) else '{"type": "finish"}'
            out.append(_strip_prefix(text, r.prefix))
        return out

    def score(self, reqs):
        return [[0.0] * len(r.candidates) for r in reqs]


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class HFBackend(Backend):

    def __init__(self, model: str, lora: str | None = None, dtype: str = "auto", device: str | None = None,
                 load_4bit: bool = False, consult_lora: str | None = None, consult_self: bool = False):
        if consult_lora:
            raise ValueError("a consult adapter needs the vllm backend")
        self.consult_self = consult_self       # the consult is this model again (other prompt format)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        kw: dict[str, Any] = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            from .compat import bf16_ok
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok() else torch.float16)
        else:
            from .compat import bf16_ok, dtype_kw
            if dtype in ("auto", None) and torch.cuda.is_available() and not bf16_ok():
                dtype = "float16"
            kw.update(dtype_kw(getattr(torch, dtype) if dtype not in ("auto", None) else "auto"))
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not load_4bit:
            self.model.to(self.device)
        self.model.eval()

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def has_adapter(self, name):
        return name == "consult" and self.consult_self

    def generate(self, reqs):
        torch = self.torch
        outs = []
        for r in reqs:
            ids = self.tok(r.prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            gen_kw = dict(max_new_tokens=r.max_new_tokens, do_sample=r.temperature > 0,
                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            if r.temperature > 0:
                gen_kw.update(temperature=r.temperature, top_p=0.95)
                torch.manual_seed(r.seed)
            with torch.no_grad():
                o = self.model.generate(**ids, **gen_kw)
            text = self.tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
            for s in STOP:
                text = text.split(s)[0]
            outs.append(text)
        return outs

    def score(self, reqs):
        torch = self.torch
        res = []
        for r in reqs:
            base = self.tok(r.prompt, add_special_tokens=False)["input_ids"]
            cand_ids = [self.tok(r.prompt + c, add_special_tokens=False)["input_ids"] for c in r.candidates]
            m = min(_common_prefix(base, c) for c in cand_ids)
            m = max(1, m)
            row = []
            for ids in cand_ids:
                t = torch.tensor([ids], device=self.device)
                with torch.no_grad():
                    logits = self.model(t).logits[0].float()
                lp = torch.log_softmax(logits, dim=-1)
                s = sum(lp[p - 1, ids[p]].item() for p in range(m, len(ids)))
                row.append(s)
            res.append(row)
        return res


class VLLMBackend(Backend):

    def __init__(self, model: str, lora: str | None = None, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.85, seed: int = 0, dtype: str = "auto",
                 enforce_eager: bool = False, max_lora_rank: int = 64, max_num_seqs: int = 64,
                 chat_kwargs: dict | None = None, consult_lora: str | None = None, consult_self: bool = False):
        from vllm import LLM
        from .compat import bf16_ok, gpu_capability
        # T4 (sm75) has no bf16
        if dtype == "auto" and gpu_capability() is not None and not bf16_ok():
            dtype = "half"
            print("[vllm] no bf16 on this GPU, using float16")
        kw: dict[str, Any] = dict(model=model, tokenizer=model, max_model_len=max_model_len,
                                  gpu_memory_utilization=gpu_memory_utilization, seed=seed, dtype=dtype,
                                  enable_prefix_caching=True, enforce_eager=enforce_eager,
                                  max_num_seqs=max_num_seqs)
        if lora or consult_lora:
            kw.update(enable_lora=True, max_lora_rank=max_lora_rank)
            if lora and consult_lora:
                kw.update(max_loras=2)         # both adapters in one batch, not one after the other
        if _is_multimodal(model):              # e.g. a Qwen3.5 teacher: text prompts only, no vision budget
            kw.update(limit_mm_per_prompt={"image": 0, "video": 0})
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.chat_kwargs = dict(chat_kwargs or {})
        self.lora_req = self.consult_req = None
        self.consult_self = bool(consult_self) and not consult_lora   # the consult is this model again
        if lora or consult_lora:
            from vllm.lora.request import LoRARequest
            if lora:
                self.lora_req = LoRARequest("adapter", 1, lora)
            if consult_lora:                   # same base model, another adapter (AgentConfig.consult)
                self.consult_req = LoRARequest("consult", 2, consult_lora)

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **self.chat_kwargs)

    def has_adapter(self, name):
        return name == "consult" and (getattr(self, "consult_self", False)
                                      or getattr(self, "consult_req", None) is not None)

    def _lora(self, r: GenRequest):
        if r.adapter == "consult" and not getattr(self, "consult_self", False):
            return getattr(self, "consult_req", None)
        return self.lora_req

    def generate(self, reqs):
        from vllm import SamplingParams
        # vLLM rejects the whole call if one prompt does not fit: such a prompt gets an empty answer instead
        # (the agent then counts a bad action), and the others get at most the room that is left
        texts, idx, params = [""] * len(reqs), [], []
        for i, r in enumerate(reqs):
            room = self.max_model_len - len(self.tok.encode(r.prompt, add_special_tokens=False))
            if room < 16:
                print(f"[vllm] prompt too long ({self.max_model_len - room} tokens), skipped", flush=True)
                continue
            idx.append(i)
            params.append(SamplingParams(temperature=r.temperature, top_p=0.95 if r.temperature > 0 else 1.0,
                                         max_tokens=min(r.max_new_tokens, room), stop=STOP, seed=r.seed))
        if idx:
            prompts = [reqs[i].prompt for i in idx]
            loras = [self._lora(reqs[i]) for i in idx]
            one = all(x is loras[0] for x in loras)          # one adapter (or none) for all: passed as before
            # candidates of one step share their prompt: one token for each such prompt first, so the cache holds
            # it and the candidates do not each compute it again (the wait is the same prompt k times otherwise)
            seen, primed, twice, twice_lora = set(), set(), [], []
            for q, lr in zip(prompts, loras):
                key = (q, id(lr))                             # the cache is kept per adapter
                if key in seen and key not in primed:
                    primed.add(key)
                    twice.append(q)
                    twice_lora.append(lr)
                seen.add(key)
            if twice:
                self.llm.generate(twice, SamplingParams(temperature=0.0, max_tokens=1),
                                  lora_request=twice_lora[0] if one else twice_lora, use_tqdm=False)
            outs = self.llm.generate(prompts, params, lora_request=loras[0] if one else loras, use_tqdm=False)
            for i, o in zip(idx, outs):
                texts[i] = o.outputs[0].text
        return texts

    def score(self, reqs):
        from vllm import SamplingParams
        prompts, index = [], []
        for k, r in enumerate(reqs):
            base = self.tok.encode(r.prompt, add_special_tokens=False)
            cand = [self.tok.encode(r.prompt + c, add_special_tokens=False) for c in r.candidates]
            m = max(1, min(_common_prefix(base, c) for c in cand))
            for j, ids in enumerate(cand):
                prompts.append({"prompt_token_ids": ids})
                index.append((k, j, m))
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outs = self.llm.generate(prompts, sp, lora_request=self.lora_req, use_tqdm=False)
        res = [[0.0] * len(r.candidates) for r in reqs]
        for o, (k, j, m) in zip(outs, index):
            ids = o.prompt_token_ids
            s = 0.0
            for p in range(m, len(ids)):
                d = o.prompt_logprobs[p]
                s += d[ids[p]].logprob
            res[k][j] = s
        return res


def _is_multimodal(model: str) -> bool:
    import json
    from pathlib import Path
    cfg = Path(model) / "config.json"
    try:
        return "vision_config" in json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def make_backend(name: str, **kw) -> Backend:
    if name == "vllm":
        return VLLMBackend(**kw)
    if name == "hf":
        return HFBackend(**kw)
    raise ValueError(f"unknown backend: {name}")


TYPE_CANDIDATES = [t + '"' for t in TYPES]
__all__ = ["Backend", "GenRequest", "ScoreRequest", "OracleBackend", "ScriptedBackend", "HFBackend",
           "VLLMBackend", "make_backend", "TYPE_PREFIX", "TYPE_CANDIDATES"]
