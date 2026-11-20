"""Check the /data layout, model folders and GPU before inference, so a bad mount fails fast."""
import sys
from pathlib import Path

sys.path.insert(0, "/code")

from lifemate.asr.transcribe import utterances  # noqa: E402
from lifemate.data import Sample, read_jsonl  # noqa: E402
from lifemate.predict import find_split  # noqa: E402
from predict import CFG  # noqa: E402


def main():
    cfg = CFG.override(sys.argv[1:])
    sp = find_split(cfg.data, cfg.split)
    rows = read_jsonl(sp.jsonl)
    print(f"[prep] split={sp.split} samples={len(rows)} dir={sp.dir}")
    if not sp.catalog.exists():
        raise SystemExit(f"[prep] missing {sp.catalog}")
    if not (sp.zaic_tools / "__init__.py").exists() and not sp.zaic_tools.with_suffix(".py").exists():
        print(f"[warn] no zaic_tools package in {sp.dir}")

    n_utt, missing = 0, []
    for r in rows:
        for _, rel in utterances(Sample(r)):
            n_utt += 1
            if not rel or not sp.audio(rel).exists():
                missing.append(f"{r['id']}:{rel}")
    print(f"[prep] audio files: {n_utt - len(missing)}/{n_utt} found")
    if missing:
        print(f"[warn] {len(missing)} missing, e.g. {missing[:5]} (those turns get empty text)")

    need = []
    if cfg.backend != "oracle":
        need.append(Path(cfg.llm) / "config.json")
    if cfg.asr != "oracle":
        need.append(Path(cfg.asr_model))
    for p in need:
        if not p.exists():
            raise SystemExit(f"[prep] missing {p}")

    fuse = Path(cfg.assets) / "fuse.json"
    if fuse.exists() and cfg.fuse != "off":
        has2 = any(Path(cfg.asr2_model).rglob("*.onnx")) if Path(cfg.asr2_model).exists() else False
        print(f"[prep] fusion: {fuse.read_text(encoding='utf-8').strip()} | second ASR "
              f"{'found' if has2 else 'MISSING (fusion off)'} at {cfg.asr2_model}")

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[prep] gpu {torch.cuda.get_device_name(0)}, {free / 2**30:.1f}/{total / 2**30:.1f} GB free, "
                  f"torch {torch.__version__}, cuda {torch.version.cuda}")
        else:
            print("[warn] no CUDA device visible (run with --gpus '\"device=0\"')")
    except ImportError:
        print("[warn] torch not installed")


if __name__ == "__main__":
    main()
