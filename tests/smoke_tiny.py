"""Smoke test of the torch code paths with a random tiny Qwen3 (no downloads)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

CHATML = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
          "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def build_tiny(work: Path, texts: list[str]) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=800, special_tokens=special,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   bos_token=None)
    fast.chat_template = CHATML
    d = work / "tiny-qwen3"
    fast.save_pretrained(d)
    cfg = Qwen3Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=16384,
                      tie_word_embeddings=True, eos_token_id=fast.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=fast.pad_token_id)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    return d


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-2500:], sep="\n")
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd[:4])}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    import make_mock
    data = work / "mock_data"
    make_mock.main(["--out", str(data)])
    py = sys.executable
    sft = work / "sft.jsonl"
    run([py, "-m", "lifemate", "build-sft", "--data", str(data), "--out", str(sft), "--n-aug", "1"])
    rows = [json.loads(x) for x in sft.read_text(encoding="utf-8").splitlines()]
    texts = [m["content"] for r in rows for m in r["messages"]] * 3
    tiny = build_tiny(work, texts)

    from transformers import AutoTokenizer
    from lifemate.train_lora import encode
    tok = AutoTokenizer.from_pretrained(tiny)
    ex = encode(tok, rows[0]["messages"], 100000)
    lab = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    want = "".join(m["content"] + "<|im_end|>" for m in rows[0]["messages"] if m["role"] == "assistant")
    assert lab.replace(" ", "") == want.replace(" ", ""), (lab[:300], want[:300])
    print("[ok] loss mask covers exactly the assistant turns")

    ad = work / "adapter"
    run([py, "-m", "lifemate", "train", "--model", str(tiny), "--data", str(sft), "--out", str(ad),
         "--max-len", "8192", "--epochs", "2", "--grad-accum", "2", "--lora-r", "4", "--lora-alpha", "8",
         "--lr", "5e-3", "--logging-steps", "2", "--dtype", "fp32"])
    merged = work / "merged"
    run([py, "-m", "lifemate", "merge-lora", "--base", str(tiny), "--adapter", str(ad), "--out", str(merged),
         "--dtype", "bf16"])
    run([py, "-m", "lifemate", "count-params", "--llm", str(merged)])

    sub = work / "hf_sub.jsonl"
    cfg = work / "cfg.json"
    cfg.write_text(json.dumps({"agent": {"max_steps": 3, "max_new_tokens": 24, "use_type_scoring": True}}),
                   encoding="utf-8")
    run([py, "-m", "lifemate", "infer", "--data", str(data), "--split", "train", "--backend", "hf",
         "--model", str(tiny), "--lora", str(ad), "--text", "oracle", "--config", str(cfg), "--limit", "3",
         "--out", str(sub), "--trace", str(work / "hf_trace.jsonl")])
    run([py, "-m", "lifemate", "score", "--data", str(data), "--split", "train", "--submission", str(sub),
         "--only-submitted", "--hash-cache", str(work / "hash.json")])
    print("[ok] smoke test passed")


if __name__ == "__main__":
    main()
