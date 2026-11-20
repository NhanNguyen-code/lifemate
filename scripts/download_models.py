"""Download models with huggingface_hub (no CLI launcher needed)."""
from __future__ import annotations

import argparse
from pathlib import Path

MODELS = {
    "llm": ("Qwen/Qwen3-4B-Instruct-2507", "base-llm"),
    "llm35": ("Qwen/Qwen3.5-4B", "base-llm35"),
    "gipformer": ("g-group-ai-lab/gipformer1.5-68M-rnnt", "gipformer"),
    "phowhisper": ("vinai/PhoWhisper-large", "phowhisper-large"),
    "phowhisper-medium": ("vinai/PhoWhisper-medium", "phowhisper-medium"),
    # teachers for `rationale` (training data only, never in the image)
    "teacher": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "teacher"),
    "teacher-small": ("Qwen/Qwen2.5-14B-Instruct", "teacher-small"),
    # Qwen3.5 (Feb 2026, Apache-2.0), FP8 ~35 GB: fits one 96 GB G4 with room for the KV cache
    "teacher35": ("Qwen/Qwen3.5-35B-A3B-FP8", "teacher35"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["llm", "gipformer"], help=" | ".join(MODELS) + " | all")
    ap.add_argument("--dir", default="models")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    keys = ["llm", "gipformer", "phowhisper"] if a.names == ["all"] else a.names
    for k in keys:
        repo, sub = MODELS[k]
        out = Path(a.dir) / sub
        print(f"downloading {repo} -> {out}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(out))
        files = sorted(p.name for p in out.rglob("*") if p.is_file() and ".cache" not in p.parts)
        print(f"  {len(files)} files: {files[:12]}{' ...' if len(files) > 12 else ''}")
        if k == "gipformer" and not any(f.endswith(".onnx") for f in files):
            print("  [warn] no .onnx files: export to sherpa-onnx before using --asr gipformer")


if __name__ == "__main__":
    main()
