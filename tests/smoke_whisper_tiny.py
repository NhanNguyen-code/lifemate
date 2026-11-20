"""Smoke test of the Whisper code paths with a random tiny Whisper (no downloads, CPU, ~1 min):
LoRA fine-tune with history prompts on the mock data, prompted / plain decoding, merge with decoding settings."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))



def special_tokens() -> list[str]:
    # the real layout: transformers finds <|vi|> by its offset after <|startoftranscript|>
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    return (["<|endoftext|>", "<|startoftranscript|>"] + [f"<|{c}|>" for c in LANGUAGES]
            + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>",
               "<|notimestamps|>"])


def build_tiny(work: Path, texts: list[str]) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import (GenerationConfig, WhisperConfig, WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer)
    bpe = Tokenizer(models.BPE(unk_token=None))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=[],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tdir = work / "tok"
    tdir.mkdir(parents=True, exist_ok=True)
    bpe.model.save(str(tdir))                                  # vocab.json + merges.txt
    tok = WhisperTokenizer(vocab_file=str(tdir / "vocab.json"), merges_file=str(tdir / "merges.txt")) \
        if "vocab_file" in WhisperTokenizer.__init__.__code__.co_varnames else \
        WhisperTokenizer(vocab=json.loads((tdir / "vocab.json").read_text()),
                         merges=[tuple(x.split()) for x in (tdir / "merges.txt").read_text().splitlines()
                                 if x and not x.startswith("#")])
    special = special_tokens()
    tok.add_special_tokens({"additional_special_tokens": special})
    ids = {t: tok.convert_tokens_to_ids(t) for t in special}
    assert all(isinstance(v, int) and v >= 0 for v in ids.values()), ids
    d = work / "tiny-whisper"
    cfg = WhisperConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                        decoder_ffn_dim=64, num_mel_bins=80, max_source_positions=1500, max_target_positions=448,
                        decoder_start_token_id=ids["<|startoftranscript|>"], pad_token_id=ids["<|endoftext|>"],
                        bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|endoftext|>"],
                        suppress_tokens=[], begin_suppress_tokens=[])
    m = WhisperForConditionalGeneration(cfg)
    g = GenerationConfig.from_model_config(cfg)
    g.decoder_start_token_id = ids["<|startoftranscript|>"]
    g.lang_to_id = {"<|en|>": ids["<|en|>"], "<|vi|>": ids["<|vi|>"]}
    g.task_to_id = {"transcribe": ids["<|transcribe|>"], "translate": ids["<|translate|>"]}
    g.no_timestamps_token_id = ids["<|notimestamps|>"]
    g.prev_sot_token_id = ids["<|startofprev|>"]
    g.is_multilingual = True
    g.forced_decoder_ids = None
    g.suppress_tokens, g.begin_suppress_tokens = [], []
    g.max_length = 448
    g._from_model_config = False             # else from_pretrained rebuilds it from the model config
    m.generation_config = g
    m.save_pretrained(d)
    WhisperProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=80), tokenizer=tok).save_pretrained(d)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    import make_mock
    from lifemate.asr.context import SETTINGS_FILE, prompt_for, read_settings
    from lifemate.asr.engines import Whisper
    from lifemate.asr.finetune_whisper import main as finetune
    from lifemate.asr.merge_whisper import main as merge
    from lifemate.data import load_samples, resolve_split

    work = Path(a.work or tempfile.mkdtemp(prefix="lm_whisper_"))
    data = work / "data"
    make_mock.main(["--out", str(data)])
    sp = resolve_split(data, "train")
    samples = load_samples(sp)
    texts = [s.audio_text or "" for s in samples] + [prompt_for(s) for s in samples] + ["Dạ em nhắn Zalo rồi ạ."]
    base = build_tiny(work, texts * 20)

    out = work / "lora"
    finetune(["--data", str(data), "--model", str(base), "--out", str(out), "--epochs", "1", "--batch-size", "2",
              "--grad-accum", "1", "--lora-r", "4", "--p-prompt", "1.0", "--keep-checkpoints", "0"])
    assert read_settings(out) == {"prompt_final": True}, read_settings(out)

    w = Whisper(str(base), lora=str(out), device="cpu", num_beams=2)
    with_hist = next(s for s in samples if s.history)
    p = prompt_for(with_hist)
    assert p, "mock sample with history should get a prompt"
    paths = [str(sp.audio(s.audio)) for s in samples[:3]] + [str(sp.audio(with_hist.audio))]
    plain = w.transcribe(paths, batch_size=2)
    mixed = w.transcribe(paths, batch_size=2, prompts=[None, None, None, p])
    assert len(plain) == len(mixed) == 4 and all(isinstance(t, str) for t in mixed)
    assert plain[:3] == mixed[:3], "unprompted rows must not change"
    assert not w._warned, "prompted decoding fell back to plain decoding"
    raw = w.model.generate(w.proc.feature_extractor([__import__("lifemate.asr.audio", fromlist=["x"]).load_audio(
        paths[-1])], sampling_rate=16000, return_tensors="pt").input_features, language="vi", task="transcribe",
        prompt_ids=w.prompt_ids(p), max_new_tokens=20)
    print("raw generate output with prompt starts with:", [int(x) for x in raw[0][:6]],
          "| prompt starts with:", [int(x) for x in w.prompt_ids(p)[:6]])
    print("prompted transcript:", repr(mixed[3][:80]))

    merged = work / "merged"
    merge(["--base", str(base), "--adapter", str(out), "--out", str(merged), "--dtype", "fp32"])
    assert (merged / SETTINGS_FILE).exists()
    w2 = Whisper(str(merged), device="cpu")
    assert len(w2.transcribe(paths[:1], prompts=[p])) == 1
    print("[smoke-whisper] ok:", work)


if __name__ == "__main__":
    main()
