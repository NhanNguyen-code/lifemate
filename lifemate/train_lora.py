"""LoRA SFT on build-sft output. Loss on assistant turns only."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate train")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, nargs="+", help="one or more jsonl files of {messages, meta}")
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA, for 12-16GB GPUs / T4")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save every epoch")
    ap.add_argument("--max-rows", type=int, default=0, help="use only the first N rows (smoke tests)")
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for the latest one in --out")
    ap.add_argument("--attn", default=None, help="e.g. flash_attention_2 / sdpa")
    return ap.parse_args(argv)


def assistant_char_spans(tok, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    full = tok.apply_chat_template(messages, tokenize=False)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        pre = tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(messages[: i + 1], tokenize=False)
        if full.startswith(upto) and upto.startswith(pre):
            end = len(upto.rstrip("\n"))
            spans.append((len(pre), end))
            cursor = end
        else:
            k = full.find(m["content"], cursor)
            if k < 0:
                raise ValueError("cannot locate an assistant turn in the rendered chat template")
            end = k + len(m["content"])
            eos = tok.eos_token or ""
            if eos and full.startswith(eos, end):
                end += len(eos)
            spans.append((k, end))
            cursor = end
    return full, spans


def encode(tok, messages: list[dict], max_len: int):
    full, spans = assistant_char_spans(tok, messages)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    j = 0
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if e <= s:
            continue
        while j < len(spans) and spans[j][1] <= s:
            j += 1
        if j < len(spans) and s >= spans[j][0] and e <= spans[j][1]:
            labels[t] = ids[t]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def load_rows(paths, max_rows=0):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if max_rows:
        rows = rows[:max_rows]
    return rows


def main(argv=None):
    a = parse(argv)
    from .compat import bf16_ok, resolve_resume, single_gpu
    single_gpu()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    rows = load_rows(a.data, a.max_rows)
    from .prompt import rows_use_thoughts, write_model_prompt
    thoughts, n_th = rows_use_thoughts(rows)
    if 0 < n_th < len(rows):
        print(f"[warn] {n_th}/{len(rows)} rows ask for thoughts: mixed prompt formats", flush=True)
    write_model_prompt(a.out, {"thoughts": thoughts})
    print(f"[data] thoughts: {thoughts}", flush=True)
    ds, skipped = [], 0
    for r in rows:
        ex = encode(tok, r["messages"], a.max_len)
        if ex is None:
            skipped += 1
        else:
            ds.append(ex)
    n_tok = sum(len(x["input_ids"]) for x in ds)
    n_lab = sum(sum(1 for y in x["labels"] if y != -100) for x in ds)
    print(f"[data] {len(ds)} examples ({skipped} too long or unlabelled), {n_tok / 1e6:.2f}M tokens, "
          f"{n_lab / 1e3:.1f}K loss tokens, {time.time() - t0:.0f}s", flush=True)
    eval_ds = None
    if a.eval_data:
        eval_ds = [x for x in (encode(tok, r["messages"], a.max_len) for r in load_rows([a.eval_data])) if x]

    cuda = torch.cuda.is_available()
    dtype_name = a.dtype or ("bf16" if bf16_ok() else ("fp16" if cuda else "fp32"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    from .compat import dtype_kw
    kw = dtype_kw(dtype)
    if a.attn:
        kw["attn_implementation"] = a.attn
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.config.use_cache = False
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not a.no_grad_ckpt)
    elif not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=[t for t in a.target_modules.split(",") if t])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    class Collator:
        def __call__(self, feats):
            m = max(len(f["input_ids"]) for f in feats)
            pad = tok.pad_token_id
            ids = [f["input_ids"] + [pad] * (m - len(f["input_ids"])) for f in feats]
            lab = [f["labels"] + [-100] * (m - len(f["labels"])) for f in feats]
            att = [[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in feats]
            return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

    class SFTTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            model = getattr(model, "module", model)   # unwrap DataParallel
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            decoder = getattr(base, "model", None)
            head = base.get_output_embeddings()
            if decoder is None or head is None:
                out = model(**inputs, labels=labels)
                return (out.loss, out) if return_outputs else out.loss
            # project only labelled positions to the vocab, not the full [seq x vocab] logits
            h = decoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        use_cache=False).last_hidden_state
            sl = labels[:, 1:]
            mask = sl != -100
            logits = head(h[:, :-1, :][mask]).float()
            loss = F.cross_entropy(logits, sl[mask], reduction="mean")
            return (loss, None) if return_outputs else loss

    steps_per_epoch = math.ceil(len(ds) / (a.batch_size * a.grad_accum))
    from .compat import training_args
    targs = training_args(
        TrainingArguments, warmup=a.warmup,
        output_dir=a.out, per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine",
        weight_decay=a.weight_decay, logging_steps=a.logging_steps,
        save_strategy="steps" if a.save_steps else "epoch", save_steps=a.save_steps or 500,
        save_total_limit=3, bf16=dtype_name == "bf16", fp16=dtype_name == "fp16",
        gradient_checkpointing=False, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, seed=a.seed, optim="paged_adamw_8bit" if a.load_4bit else "adamw_torch",
        eval_strategy="epoch" if eval_ds else "no", per_device_eval_batch_size=1,
        use_cpu=not cuda)
    print(f"[train] {steps_per_epoch} steps/epoch, dtype={dtype_name}, 4bit={a.load_4bit}", flush=True)
    from transformers import TrainerCallback

    class KeepIntervals(TrainerCallback):
        # a resumed run takes save/log intervals from trainer_state.json; use the ones given now (GPU switched)
        def on_train_begin(self, args, state, control, **kw):
            if args.save_strategy in ("steps", getattr(args.save_strategy, "STEPS", None)) and args.save_steps >= 1:
                state.save_steps = int(args.save_steps)
            if args.logging_steps >= 1:
                state.logging_steps = int(args.logging_steps)

    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds, data_collator=Collator(),
                         callbacks=[KeepIntervals()])
    trainer.train(resume_from_checkpoint=resolve_resume(a.out, a.resume))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "train_args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[train] adapter saved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
