"""LoRA fine-tune of PhoWhisper on train audio (final request + history turns)."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def decoder_lists(labels: list[list[int]], prompts: list[list[int]], sot: int, pad: int,
                  max_len: int = 448) -> tuple[list[list[int]], list[list[int]]]:
    """Decoder inputs and targets, right padded. labels: tokenizer output (<|sot|> <|vi|> ... text <|eot|>);
    prompts: [] or [<|startofprev|>, ...] as Whisper's generate() puts it in front of <|sot|> at inference.
    Prompt tokens and <|sot|> are never targets."""
    dec, tgt = [], []
    for lab, pr in zip(labels, prompts):
        lab = list(lab)
        if lab and lab[0] == sot:
            lab = lab[1:]
        pr = list(pr or [])
        over = len(pr) + 1 + len(lab) - max_len
        if pr and over > 0:                       # long transcript: drop prompt tokens (keep <|startofprev|>)
            pr = pr[:1] + pr[1 + over:] if len(pr) - over > 1 else []
        full = pr + [sot] + lab
        d, t = full[:-1], full[1:]
        n_mask = len(pr)                          # targets P1..Pn and <|sot|>
        dec.append(d)
        tgt.append([-100] * n_mask + t[n_mask:])
    m = max(len(d) for d in dec)
    dec = [d + [pad] * (m - len(d)) for d in dec]
    tgt = [t + [-100] * (m - len(t)) for t in tgt]
    return dec, tgt


def decoder_pairs(labels, prompts, sot: int, pad: int):
    import torch
    dec, tgt = decoder_lists(labels, prompts, sot, pad)
    return torch.tensor(dec, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate finetune-whisper")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--val-ids", default=None, help="keep these samples' audio out of training (honest end-to-end dev)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-utts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (use on T4)")
    ap.add_argument("--resume", default=None, help="'auto' resumes from the latest checkpoint in --out")
    ap.add_argument("--p-prompt", type=float, default=0.0,
                    help="share of utterances trained with their history context as Whisper prompt (asr/context.py)")
    ap.add_argument("--keep-checkpoints", type=int, default=2, help="epoch checkpoints to keep (0 = all)")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    from ..data import load_samples, read_ids, resolve_split
    from .audio import SR, load_audio
    from .context import clip_prompt_ids, prompt_for, write_settings
    from .transcribe import utterances

    sp = resolve_split(a.data, "train")
    samples = load_samples(sp)
    val = set(read_ids(a.val_ids) or [])
    items = []
    for s in samples:
        f, h = s.transcripts()
        texts = [f or ""] + [x or "" for x in h]
        for (key, rel), txt in zip(utterances(s), texts):
            if txt.strip():
                # the context this turn would get: history before it (all of it for the final request)
                ctx = prompt_for(s, None if key == "final" else int(key[1:])) if a.p_prompt > 0 else ""
                items.append({"id": s.id, "path": str(sp.audio(rel)), "text": txt.strip(), "val": s.id in val,
                              "prompt": ctx})
    train = [x for x in items if not x["val"]]
    dev = [x for x in items if x["val"]][:300]
    random.Random(a.seed).shuffle(train)
    if a.max_utts:
        train = train[: a.max_utts]
    n_ctx = sum(1 for x in train if x["prompt"])
    print(f"[asr-ft] train {len(train)} utterances ({n_ctx} with history context, p_prompt={a.p_prompt}), "
          f"dev {len(dev)}")

    proc = WhisperProcessor.from_pretrained(a.model)
    cuda = torch.cuda.is_available()
    from ..compat import bf16_ok, dtype_kw, resolve_resume
    bf16 = bf16_ok()
    model = WhisperForConditionalGeneration.from_pretrained(
        a.model, **dtype_kw(torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    try:
        model.generation_config.language = "vi"
        model.generation_config.task = "transcribe"
    except Exception:
        pass
    if a.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    prefix = proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>"])[0]
    pad_id = proc.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows, p_prompt: float = 0.0):
            self.rows = rows
            self.p_prompt = p_prompt

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            feats = proc.feature_extractor(load_audio(r["path"]), sampling_rate=SR).input_features[0]
            proc.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
            labels = proc.tokenizer(r["text"]).input_ids
            prompt = []
            # random per epoch (DataLoader seeds `random` in each worker): a turn is seen with and without context
            if r.get("prompt") and self.p_prompt > 0 and random.random() < self.p_prompt:
                prompt = clip_prompt_ids(list(proc.tokenizer.get_prompt_ids(r["prompt"], return_tensors="np")))
            return {"input_features": feats, "labels": labels, "prompt": prompt}

    def collate(batch):
        feats = torch.from_numpy(np.stack([np.asarray(b["input_features"], dtype=np.float32) for b in batch]))
        dec, tgt = decoder_pairs([b["labels"] for b in batch], [b.get("prompt") or [] for b in batch],
                                 prefix, pad_id)
        return {"input_features": feats.to(model.dtype if hasattr(model, "dtype") else torch.float32),
                "decoder_input_ids": dec, "labels": tgt}

    from ..compat import training_args
    args = training_args(
        Seq2SeqTrainingArguments, warmup=0.05,
        output_dir=a.out, per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine",
        logging_steps=20, save_strategy="epoch", save_total_limit=a.keep_checkpoints or None,
        bf16=bf16, fp16=cuda and not bf16,
        eval_strategy="epoch" if dev else "no", per_device_eval_batch_size=a.batch_size,
        report_to="none", remove_unused_columns=False, label_names=["labels"], seed=a.seed,
        dataloader_num_workers=0 if os.name == "nt" else 2, use_cpu=not cuda)
    # dev loss without prompts: comparable across runs
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train, a.p_prompt),
                        eval_dataset=DS(dev) if dev else None, data_collator=collate)
    tr.train(resume_from_checkpoint=resolve_resume(a.out, a.resume))
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)
    (Path(a.out) / "asr_ft_args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    # predict / transcribe read this: a model trained with context gets it on the final request
    write_settings(a.out, prompt_final=a.p_prompt > 0)
    print(f"[asr-ft] adapter saved to {a.out}")


if __name__ == "__main__":
    main()
