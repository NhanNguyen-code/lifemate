"""Dev splits. Leave-apps-out is the one to trust: public/private contain apps unseen in train."""
from __future__ import annotations

import random
from collections import Counter

from .data import Sample, gold_category


def used_apps(s: Sample, meta: set[str]) -> set[str]:
    return {c["name"].split(".")[0] for c in (s.gold_calls or []) if c["name"] not in meta}


def leave_apps_out(samples: list[Sample], catalog, n_apps: int = 10, seed: int = 0,
                   oos_frac: float = 0.07) -> dict:
    meta = catalog.meta_tools
    counts = Counter(a for s in samples for a in used_apps(s, meta))
    apps = sorted(counts)
    rng = random.Random(seed)
    rng.shuffle(apps)
    held = set(apps[:n_apps])
    val = {s.id for s in samples if used_apps(s, meta) & held}
    oos = sorted(s.id for s in samples if gold_category(s, meta) == "oos")
    rng.shuffle(oos)
    k = int(round(oos_frac * len(val)))
    val |= set(oos[:k])
    return {"type": "leave_apps_out", "seed": seed, "held_apps": sorted(held),
            "train_ids": [s.id for s in samples if s.id not in val],
            "val_ids": [s.id for s in samples if s.id in val]}


def random_split(samples: list[Sample], n_val: int = 200, seed: int = 0) -> dict:
    ids = [s.id for s in samples]
    rng = random.Random(seed)
    rng.shuffle(ids)
    val = set(ids[:n_val])
    return {"type": "random", "seed": seed, "held_apps": [],
            "train_ids": [s.id for s in samples if s.id not in val],
            "val_ids": [s.id for s in samples if s.id in val]}


def kfold(ids: list[str], k: int = 5, seed: int = 0) -> list[list[str]]:
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return [ids[i::k] for i in range(k)]
