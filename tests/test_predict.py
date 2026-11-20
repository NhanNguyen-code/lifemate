"""Tests for the docker inference path (lifemate/predict.py) on mock data laid out like BTC's /data."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import make_mock
from lifemate import predict as P
from lifemate.catalog import Catalog
from lifemate.compare import compare
from lifemate.data import load_samples, make_env, read_jsonl
from lifemate.scorer import Scorer

TMP = Path(tempfile.mkdtemp(prefix="lm_predict_"))
APPS = ["contacts", "sms", "smart_home", "calendar", "banking"]


def setUpModule():
    make_mock.main(["--out", str(TMP / "dev")])                     # train/ + public_test/
    make_mock.write_split(TMP / "btc", "private_test", make_mock.TRAIN, APPS, True)   # labelled copy
    with redirect_stdout(io.StringIO()):
        from lifemate.cli import main as cli
        cli(["build-assets", "--data", str(TMP / "dev"), "--out", str(TMP / "assets")])


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def cfg(**kw) -> P.PredictConfig:
    base = dict(data=str(TMP / "btc"), backend="oracle", asr="oracle", assets=str(TMP / "assets"),
                config=str(ROOT / "configs" / "default.json"), bias=None, warmup=False,
                out=str(TMP / "result" / "submission.jsonl"))
    base.update(kw)
    return P.PredictConfig(**base)


class TestPredict(unittest.TestCase):
    def test_find_split_layouts(self):
        sp = P.find_split(str(TMP / "btc"))
        self.assertEqual((sp.split, sp.root), ("private_test", TMP / "btc"))
        self.assertEqual(P.find_split(str(TMP / "dev")).split, "public_test")
        self.assertEqual(P.find_split(str(TMP / "dev"), "train").split, "train")
        sp2 = P.find_split(str(TMP / "btc" / "private_test"))
        self.assertEqual(sp2.split, "private_test")
        self.assertTrue(sp2.jsonl.exists())

    def test_rows_have_time_and_score_100(self):
        with redirect_stdout(io.StringIO()) as out:
            rows = P.run(cfg())
        log = out.getvalue()
        self.assertIn("[time] model loading", log)
        self.assertIn("[time] prediction", log)
        sp = P.find_split(str(TMP / "btc"))
        want_ids = [r["id"] for r in read_jsonl(sp.jsonl)]
        written = read_jsonl(TMP / "result" / "submission.jsonl")
        self.assertEqual([r["id"] for r in written], want_ids)
        for r in written:
            self.assertEqual(list(r), ["id", "tool_calls", "state_hash", "time"])
            self.assertIsInstance(r["time"], int)
            self.assertGreaterEqual(r["time"], 0)
        self.assertEqual(written, json.loads(json.dumps(rows)))
        cat = Catalog.load(sp.catalog)
        rep = Scorer(sp, cat).score(written, load_samples(sp))
        self.assertEqual(rep["score"], 100.0)

    def test_notebook_loop_matches_predict(self):
        c = cfg(out=str(TMP / "result" / "a.jsonl"))
        with redirect_stdout(io.StringIO()):
            rows = P.run(c)
        pr = P.Predictor(c)
        sp = P.find_split(c.data, c.split)
        pr.open_split(sp)
        nb = []
        for item in read_jsonl(sp.jsonl):
            calls, h = pr.run_one(item)
            nb.append({"id": item["id"], "tool_calls": calls, "state_hash": h, "time": 1})
        path = TMP / "result" / "jupyter_submission.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in nb), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(compare(c.out, str(path)), 0)
            nb[0]["state_hash"] = "x"
            path.write_text("".join(json.dumps(r) + "\n" for r in nb), encoding="utf-8")
            self.assertEqual(compare(c.out, str(path)), 1)
        self.assertEqual(len(rows), len(nb))

    def test_failure_keeps_one_toolenv(self):
        c = cfg()
        pr = P.Predictor(c)
        sp = P.find_split(c.data)
        pr.open_split(sp)
        made = []
        real = P.make_env

        def counting(sp_, sid):
            made.append(sid)
            return real(sp_, sid)

        def broken(_s):
            raise RuntimeError("asr down")

        P.make_env, pr.transcribe = counting, broken
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                row = pr.predict(read_jsonl(sp.jsonl)[0])
        finally:
            P.make_env = real
        self.assertEqual(made, [row["id"]])
        self.assertEqual(row["tool_calls"], [])
        self.assertEqual(row["state_hash"], make_env(sp, row["id"]).get_state_hash())

    def test_override(self):
        c = P.PredictConfig().override(["--limit", "5", "--no-warmup", "--gpu-mem", "0.5", "--split", "public_test"])
        self.assertEqual((c.limit, c.warmup, c.gpu_mem, c.split, c.out), (5, False, 0.5, "public_test",
                                                                         "/result/submission.jsonl"))


if __name__ == "__main__":
    unittest.main()
