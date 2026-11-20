"""Load audio as 16 kHz mono float32."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    try:
        import soundfile as sf
        x, r = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            r = w.getframerate()
            ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dt = {1: np.int8, 2: np.int16, 4: np.int32}[width]
        x = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
    if r != sr:
        x = resample(x, r, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    try:
        import soxr                     # same as librosa's default (soxr_hq), without numba
        return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)
    except ImportError:
        n = int(round(len(x) * sr_out / sr_in))
        if n <= 1 or len(x) <= 1:
            return x
        t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        t_out = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)
