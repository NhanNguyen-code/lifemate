"""ASR engines: gipformer (zipformer RNN-T via sherpa-onnx) and PhoWhisper (transformers, optional LoRA)."""
from __future__ import annotations

from pathlib import Path

from .audio import SR, load_audio


def _pick(d: Path, key: str, int8: bool) -> str:
    cands = sorted(d.rglob(f"*{key}*.onnx"))
    if not cands:
        raise FileNotFoundError(f"no *{key}*.onnx in {d}")
    pref = [c for c in cands if ("int8" in c.name) == int8]
    return str((pref or cands)[0])


class Gipformer:

    def __init__(self, model_dir: str, device: str = "cpu", num_threads: int = 4, int8: bool = False,
                 decoding: str = "modified_beam_search", max_active_paths: int = 4):
        import sherpa_onnx
        d = Path(model_dir)
        tokens = next(iter(sorted(d.rglob("tokens.txt"))), None)
        if tokens is None:
            raise FileNotFoundError(f"no tokens.txt in {d}")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_pick(d, "encoder", int8), decoder=_pick(d, "decoder", int8), joiner=_pick(d, "joiner", int8),
            tokens=str(tokens), num_threads=num_threads, sample_rate=SR, feature_dim=80,
            decoding_method=decoding, max_active_paths=max_active_paths,
            provider="cuda" if device.startswith("cuda") else "cpu")

    def transcribe(self, paths: list[str], batch_size: int = 16, prompts=None) -> list[str]:
        out = []                               # transducer: no text prompt, `prompts` is ignored
        for i in range(0, len(paths), batch_size):
            streams = []
            for p in paths[i:i + batch_size]:
                s = self.rec.create_stream()
                s.accept_waveform(SR, load_audio(p))
                streams.append(s)
            self.rec.decode_streams(streams)
            out += [s.result.text.strip().lower() for s in streams]
        return out


class Whisper:

    def __init__(self, model: str, lora: str | None = None, device: str = "cuda", num_beams: int = 1):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.proc = WhisperProcessor.from_pretrained(lora if lora and (Path(lora) / "preprocessor_config.json")
                                                     .exists() else model)
        from ..compat import dtype_kw
        m = WhisperForConditionalGeneration.from_pretrained(model, **dtype_kw(self.dtype))
        if lora:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, lora).merge_and_unload()
        self.model = m.to(self.device).eval()
        self.num_beams = num_beams
        self.sot = self.proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self._warned = False

    def prompt_ids(self, text: str):
        from .context import clip_prompt_ids
        ids = self.proc.get_prompt_ids(text, return_tensors="np")
        return self.torch.tensor(clip_prompt_ids(list(ids)), dtype=self.torch.long, device=self.device)

    def _strip(self, seq) -> list[int]:
        # some transformers versions return the prompt in front of the transcript: keep what follows <|sot|>
        ids = [int(x) for x in seq]
        if self.sot in ids:
            ids = ids[len(ids) - ids[::-1].index(self.sot):]
        return ids

    def _generate(self, arrs: list, prompt: str | None = None) -> list[str]:
        torch = self.torch
        feats = self.proc.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        kw = {"num_beams": self.num_beams, "max_new_tokens": 200}
        if prompt:
            kw["prompt_ids"] = self.prompt_ids(prompt)
        with torch.no_grad():
            try:
                gen = self.model.generate(feats, language="vi", task="transcribe", **kw)
            except (TypeError, ValueError):   # generation_config without lang_to_id
                gen = self.model.generate(feats, **kw)
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        return [self.proc.tokenizer.decode(self._strip(s), skip_special_tokens=True).strip() for s in seqs]

    def transcribe(self, paths: list[str], batch_size: int = 16, prompts: list[str | None] | None = None) -> list[str]:
        """prompts[i]: context text for paths[i] (decoded on its own, transformers takes one prompt per call)."""
        out = [""] * len(paths)
        plain = [i for i in range(len(paths)) if not (prompts and prompts[i])]
        for j in range(0, len(plain), batch_size):
            idx = plain[j:j + batch_size]
            for i, t in zip(idx, self._generate([load_audio(paths[i]) for i in idx])):
                out[i] = t
        done = set(plain)
        for i in range(len(paths)):
            if i in done:
                continue
            arr = load_audio(paths[i])
            try:
                out[i] = self._generate([arr], prompts[i])[0]
            except Exception as ex:           # a prompt must never cost the transcript itself
                if not self._warned:
                    print(f"[asr] prompted decoding failed ({type(ex).__name__}: {ex}); without prompt", flush=True)
                    self._warned = True
                out[i] = self._generate([arr])[0]
        return out


def detect_kind(model_dir: str) -> str:
    d = Path(model_dir)
    if any(d.rglob("*.onnx")) and any(d.rglob("tokens.txt")):
        return "gipformer"
    cfg = d / "config.json"
    if cfg.exists() and "whisper" in cfg.read_text(encoding="utf-8").lower():
        return "whisper"
    raise ValueError(f"cannot tell ASR type of {d}: expected onnx + tokens.txt (gipformer) or a whisper config")


def make_engine(kind: str, model: str, lora: str | None = None, device: str = "cuda", num_beams: int = 1):
    if kind == "auto":
        kind = detect_kind(model)
    if kind == "gipformer":
        import os
        # pip sherpa-onnx is CPU-only; 68M params is fast enough there
        return Gipformer(model, device="cpu", num_threads=max(1, min(16, os.cpu_count() or 4)))
    if kind == "whisper":
        return Whisper(model, lora=lora, device=device, num_beams=num_beams)
    raise ValueError(kind)
