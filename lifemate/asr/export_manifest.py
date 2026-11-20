"""Kaldi-style data dirs (wav.scp / text / utt2spk) for fine-tuning gipformer with icefall."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lifemate asr-manifest")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ids", default=None)
    ap.add_argument("--lowercase", action="store_true", default=True)
    a = ap.parse_args(argv)
    from ..data import load_samples, read_ids, resolve_split
    from .transcribe import utterances
    sp = resolve_split(a.data, "train")
    val = set(read_ids(a.val_ids) or [])
    parts = {"train": [], "dev": []}
    for s in load_samples(sp):
        f, h = s.transcripts()
        texts = [f or ""] + [x or "" for x in h]
        for (key, rel), txt in zip(utterances(s), texts):
            if not txt.strip():
                continue
            uid = f"{s.id}-{key}"
            t = " ".join(txt.lower().split()) if a.lowercase else txt.strip()
            parts["dev" if s.id in val else "train"].append((uid, str(sp.audio(rel).resolve()), t, s.id))
    for name, rows in parts.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        rows.sort()
        (d / "wav.scp").write_text("".join(f"{u} {p}\n" for u, p, _, _ in rows), encoding="utf-8")
        (d / "text").write_text("".join(f"{u} {t}\n" for u, _, t, _ in rows), encoding="utf-8")
        (d / "utt2spk").write_text("".join(f"{u} {spk}\n" for u, _, _, spk in rows), encoding="utf-8")
        spk: dict[str, list[str]] = {}
        for u, _, _, sid in rows:
            spk.setdefault(sid, []).append(u)
        (d / "spk2utt").write_text("".join(f"{k} {' '.join(v)}\n" for k, v in sorted(spk.items())), encoding="utf-8")
        print(f"{name}: {len(rows)} utterances -> {d}")


if __name__ == "__main__":
    main()
