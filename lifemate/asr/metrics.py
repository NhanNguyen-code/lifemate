"""ASR metrics that matter for the agent: WER, tone-free WER, WER after ITN, exact numbers/times per utterance."""
from __future__ import annotations

import re

from ..data import Sample, Transcripts
from ..textnorm import normalize, strip_diacritics

_P = re.compile(r"[^\w\s%:]", re.UNICODE)


def _words(s: str) -> list[str]:
    return _P.sub(" ", s.lower()).split()


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def _slots(s: str) -> list[str]:
    return sorted(h for h in normalize(s).hints)


def evaluate(samples: list[Sample], tr: Transcripts, show: int = 0) -> dict:
    tot = {"words": 0, "e": 0, "e_nodiac": 0, "e_itn": 0, "utt": 0, "slot_utt": 0, "slot_ok": 0,
           "f_words": 0, "f_e": 0, "f_e_itn": 0}
    wrong = []
    for s in samples:
        ref_final, ref_hist = s.transcripts()
        if not tr.has(s.id) or ref_final is None:
            continue
        pairs = [(ref_final, tr.final(s) or "")]
        hist = tr.history(s) or []
        for k, ref in enumerate(ref_hist):
            if ref is not None and k < len(hist):
                pairs.append((ref, hist[k]))
        for n, (ref, hyp) in enumerate(pairs):
            rw, hw = _words(ref), _words(hyp)
            e, e_itn = edits(rw, hw), edits(_words(normalize(ref).text), _words(normalize(hyp).text))
            tot["words"] += len(rw)
            tot["utt"] += 1
            tot["e"] += e
            tot["e_nodiac"] += edits(_words(strip_diacritics(ref)), _words(strip_diacritics(hyp)))
            tot["e_itn"] += e_itn
            if n == 0:                         # the final request: what the agent acts on
                tot["f_words"] += len(rw)
                tot["f_e"] += e
                tot["f_e_itn"] += e_itn
            rs = _slots(ref)
            if rs:
                hs = _slots(hyp)
                tot["slot_utt"] += 1
                tot["slot_ok"] += rs == hs
                if rs != hs and len(wrong) < show:
                    wrong.append(f"{s.id} gold: {rs} | asr: {hs}")
    w, fw = max(1, tot["words"]), max(1, tot["f_words"])
    return {"utterances": tot["utt"], "words": tot["words"],
            "wer": round(100 * tot["e"] / w, 2),
            "wer_no_tones": round(100 * tot["e_nodiac"] / w, 2),
            "wer_after_itn": round(100 * tot["e_itn"] / w, 2),
            "wer_final": round(100 * tot["f_e"] / fw, 2),
            "wer_final_after_itn": round(100 * tot["f_e_itn"] / fw, 2),
            "utts_with_numbers": tot["slot_utt"],
            "numbers_all_correct": round(100 * tot["slot_ok"] / max(1, tot["slot_utt"]), 2),
            "number_mismatches": wrong}
