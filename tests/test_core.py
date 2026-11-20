"""Unit tests on the mock data: python -m unittest discover -s tests"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_mock
from lifemate.actions import canon_key, action_text, parse_action
from lifemate.agent import AgentConfig
from lifemate.catalog import Catalog
from lifemate.data import read_jsonl, Sample, Transcripts, load_samples, make_env, resolve_split
from lifemate.fewshot import FewShotPool, usable_apps
from lifemate.guard import Guard, GuardConfig, GuardState
from lifemate.llm import OracleBackend, ScriptedBackend
from lifemate.prompt import PromptConfig
from lifemate.runner import RunContext, build_episode, initial_messages, run_split
from lifemate.scorer import Scorer, max_matching, token_f1
from lifemate.sft_data import SFTConfig, build_rows
from lifemate.splits import leave_apps_out
from lifemate.textnorm import normalize
from lifemate.timeutil import FormatBook, coerce_shape

DATA = HERE / "mock_data"


def setUpModule():
    make_mock.main(["--out", str(DATA)])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = resolve_split(DATA, "train")
        cls.cat = Catalog.load(cls.sp.catalog)
        cls.samples = load_samples(cls.sp)
        cls.by_id = {s.id: s for s in cls.samples}
        cls.fb = FormatBook()
        for s in cls.samples:
            cls.fb.observe(s.gold_calls or [])

    def rows(self, results):
        return [{"id": r["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]} for r in results]


class TestTextNorm(unittest.TestCase):
    CASES = [
        ("hai trăm rưỡi nghìn nhé", "số tiền 250000"),
        ("chuyển cho hằng một triệu hai tiền ăn", "số tiền 1200000"),
        ("độ sáng năm mươi phần trăm", "phần trăm 50"),
        ("họp lúc tám giờ sáng mai", "giờ 08:00"),
        ("tối thứ sáu này bảy giờ rưỡi", "giờ 19:30"),
        ("hẹn lúc chín giờ kém mười lăm", "giờ 08:45"),
        ("số không chín không một một một một hai hai hai", "dãy số 0901111222"),
        ("chuyển hai triệu ba trăm cho mẹ", "số tiền 2300000"),
        ("mười hai giờ đêm", "giờ 00:00"),
        ("lúc 8h30 tối", "giờ 20:30"),
        ("chuyển 2,5 triệu", "số tiền 2500000"),
        ("một trăm linh năm nghìn", "số tiền 105000"),
    ]

    def test_cases(self):
        for text, hint in self.CASES:
            with self.subTest(text=text):
                self.assertIn(hint, normalize(text).hints)

    def test_no_false_numbers(self):
        for text in ("thứ năm tuần sau", "năm nay mình bận", "ba mẹ khỏe không", "hai năm nay"):
            self.assertEqual(normalize(text).hints, [], text)

    # spoken shortcuts and digit forms seen in train: ASR heard them right, ITN used to get them wrong
    SPOKEN = [
        ("chỉnh điều hoà hai tư độ", "số 24 độ"),
        ("hăm lăm độ", "số 25 độ"),
        ("hai sáu độ", "số 26 độ"),
        ("bốn lăm phút nữa", "số 45 phút"),
        ("đi mất một tiếng bốn lăm", "số 45 phút"),
        ("một trăm hai chín triệu", "số tiền 129000000"),
        ("hai mốt giờ", "giờ 21:00"),
        ("báo thức 8 giờ kém 15", "giờ 07:45"),
        ("9 giờ kém 15 tối", "giờ 20:45"),
        ("gọi 0987654321", "dãy số 0987654321"),
        ("số 0950 536 0367", "dãy số 09505360367"),
        ("không chín năm, không năm ba sáu, không ba sáu bảy", "dãy số 09505360367"),
        ("gọi số không chín tám lăm, không một hai, ba bốn năm", "dãy số 0985012345"),
        ("đặt báo thức bảy giờ kẻo mười lăm sáng", "giờ 06:45"),
        ("hẹn khách mười giờ kế hai mươi tối", "giờ 21:40"),
        ("họp lúc 8h kém 15", "giờ 07:45"),
        ("8h rưỡi tối", "giờ 20:30"),
        ("nhắc mình gọi lại lúc tám rưỡi tối mai", "giờ 20:30"),
        ("báo thức 7 rưỡi sáng mai", "giờ 07:30"),
    ]

    def test_spoken_forms(self):
        for text, hint in self.SPOKEN:
            with self.subTest(text=text):
                self.assertIn(hint, normalize(text).hints)

    def test_ranges_and_separate_numbers_stay_apart(self):
        self.assertNotIn("số 23 ngày", normalize("hai ba ngày").hints)
        self.assertFalse(any(h.startswith("dãy số") for h in normalize("chuyển 500000 2 lần").hints))
        self.assertEqual(normalize("gọi 0987654321").text, "gọi 0987654321")
        self.assertIn("giờ 08:00", normalize("tám giờ kế tiếp thì sao").hints)
        for text in ("chuyển hai triệu rưỡi", "đi mất một tiếng rưỡi", "mua ba rưỡi kí thịt"):
            self.assertFalse(any(h.startswith("giờ") for h in normalize(text).hints), text)


class TestSFTInputs(Base):
    def test_later_transcripts_win(self):
        import json
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "a.jsonl").write_text(json.dumps({"id": "x", "final": "gipformer", "history": []}) + "\n", encoding="utf-8")
        (d / "b.jsonl").write_text(json.dumps({"id": "x", "final": "phowhisper", "history": []}) + "\n", encoding="utf-8")
        self.assertEqual(Transcripts.load([d / "a.jsonl", d / "b.jsonl", d / "missing.jsonl"]).rows["x"]["final"],
                         "phowhisper")

    def test_confuse_keeps_label(self):
        from lifemate.sft_data import confuse_words
        self.assertEqual(confuse_words("Nhắc tôi uống thuốc"), "Nhắn tôi uống thuốc")
        self.assertEqual(confuse_words("đặt báo thức 6 giờ"), "đặt báo thức 6 giờ")
        rows, stat = build_rows(self.samples, self.cat, None, None, PromptConfig(fewshot_k=0),
                                SFTConfig(n_aug=0, p_asr=0.0, p_confuse=1.0))
        self.assertGreater(stat.get("aug:confuse", 0), 0)
        plain, _ = build_rows(self.samples, self.cat, None, None, PromptConfig(fewshot_k=0),
                              SFTConfig(n_aug=0, p_asr=0.0, p_confuse=0.0))
        for r, q in zip(rows, plain):       # only the user text changes, never the target turns
            self.assertEqual([m for m in r["messages"] if m["role"] == "assistant"],
                             [m for m in q["messages"] if m["role"] == "assistant"])


class TestTimeFill(Base):
    # a date-only value for a date-time parameter: take the time the user said, or send it back to the model
    def guard(self, clock):
        fb = FormatBook()
        fb.observe([{"name": "calendar.create_event", "arguments": {"start_time": "2026-10-21T08:00:00+07:00"}}])
        s = next(x for x in self.samples if "calendar" in x.raw["context"]["installed_apps"])
        return Guard(self.cat, list(self.cat.tools), s.context, GuardConfig(), fb, clock=clock), make_env(self.sp, s.id)

    def test_fills_spoken_time(self):
        g, env = self.guard(["05:00", "05:30"])
        d = g.check({"type": "write", "name": "calendar.create_event",
                     "arguments": {"title": "chạy bộ", "start_time": "2026-11-18", "end_time": "2026-11-18"}},
                    env, GuardState())
        self.assertTrue(d.action["arguments"]["start_time"].startswith("2026-11-18T05:00:00"))
        self.assertTrue(d.action["arguments"]["end_time"].startswith("2026-11-18T05:30:00"))

    def test_no_spoken_time_goes_back_to_model(self):
        g, env = self.guard([])
        d = g.check({"type": "write", "name": "calendar.create_event",
                     "arguments": {"title": "họp", "start_time": "2026-11-18"}}, env, GuardState())
        self.assertEqual(d.kind, "reject")
        self.assertIn("start_time", d.message)

    def test_date_param_stays_a_date(self):
        g, env = self.guard(["19:45"])
        d = g.check({"type": "read", "name": "calendar.list_events", "arguments": {"date": "2026-11-18"}},
                    env, GuardState())
        self.assertEqual(d.action["arguments"]["date"], "2026-11-18")
        self.assertEqual(FormatBook().expected("tickets.search", "travel_date", {}), "date")


class TestHistoryAndAskRules(Base):
    def test_long_results_drop_whole_items(self):
        from lifemate.prompt import compact_json
        res = {"ok": True, "results": [{"id": f"c_{i:03d}", "name": f"Người số {i}", "phone": f"09{i:08d}"}
                                       for i in range(10)]}
        s = compact_json(res, 400, 12)
        self.assertLessEqual(len(s), 400)
        self.assertFalse(s.endswith("…"))
        self.assertIn("còn", s)
        self.assertEqual(len(re.findall(r'"id": "c_\d{3}", "name": "[^"]+", "phone": "\d{10}"\}', s)),
                         s.count('"id"'), s)
        self.assertEqual(compact_json(res, 5000, 12).count('"id"'), 10)
        self.assertTrue(compact_json({"text": "x" * 50}, 20, 12).endswith("…"))

    def _guard(self, sid, **cfg):
        s = self.by_id[sid]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        g = Guard(self.cat, tools, s.context, GuardConfig(**cfg), self.fb)
        return s, g, g.init_state(s), make_env(self.sp, s.id)

    def test_history_read_cache(self):
        read = {"type": "read", "name": "contacts.search", "arguments": {"query": "mẹ"}}
        s, g, st, env = self._guard("train_0006")
        self.assertEqual(g.check(read, env, st).kind, "execute")
        s, g, st, env = self._guard("train_0006", cache_history_reads=True)
        d = g.check(read, env, st)
        self.assertEqual(d.kind, "cached")
        self.assertIn("c_me", json.dumps(d.result, ensure_ascii=False))
        other = {"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}
        self.assertEqual(g.check(other, env, st).kind, "execute")
        self.assertEqual(env.get_tool_calls(), [])

    def test_ask_for_an_id_before_any_read(self):
        ask = {"type": "ask", "name": "sys.ask_missing_info",
               "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}
        s, g, st, env = self._guard("train_0001", ask_ids_after_read=True)
        d = g.check(ask, env, st)
        self.assertEqual(d.kind, "reject")
        self.assertIn("contacts.search", d.message)
        g.record({"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}, {"ok": True}, st)
        self.assertEqual(g.check(ask, env, st).kind, "execute")      # searched, still unclear: asking is fine
        s, g, st, env = self._guard("train_0006", ask_ids_after_read=True)
        ask_tr = {"type": "ask", "name": "sys.ask_missing_info",
                  "arguments": {"tool_name": "banking.transfer", "params": ["contact_id"]}}
        self.assertEqual(g.check(ask_tr, env, st).kind, "execute")   # contacts read in the history
        s, g, st, env = self._guard("train_0001")
        self.assertEqual(g.check(ask, env, st).kind, "execute")      # flag off

    def test_id_from_the_wrong_app(self):
        send = {"type": "write", "name": "sms.send", "arguments": {"contact_id": "light_living", "content": "xin chào"}}
        status = {"ok": True, "devices": [{"id": "light_living", "name": "Đèn phòng khách"}]}
        for typed, want in ((False, "execute"), (True, "reject")):
            s, g, st, env = self._guard("train_0001", xref_typed=typed)
            g.record({"type": "read", "name": "smart_home.get_status", "arguments": {}}, status, st)
            d = g.check(send, env, st)
            self.assertEqual(d.kind, want, d.message)
        self.assertIn("contacts.search", d.message)
        g.record({"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}},
                 {"ok": True, "results": [{"id": "c_lan", "name": "Chị Lan"}]}, st)
        ok = {"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "xin chào"}}
        self.assertEqual(g.check(ok, env, st).kind, "execute")
        s, g, st, env = self._guard("train_0006", xref_typed=True)   # IDs from the history count for their app
        tr = {"type": "write", "name": "banking.transfer", "arguments": {"contact_id": "c_me", "amount": 250000}}
        self.assertEqual(g.check(tr, env, st).kind, "execute")

    def test_names_as_stored(self):
        from lifemate.guard import canonical_name
        fb = FormatBook()
        for q in ["Mẹ", "Cô giáo Mai", "Tuấn", "Bác Ba", "Hương"] * 5:
            fb.observe([{"name": "contacts.search", "arguments": {"query": q, "note": q.lower()}}])
        names = fb.names("contacts.search", "query")
        self.assertEqual(names, {"Mẹ", "Cô giáo Mai", "Tuấn", "Bác Ba", "Hương"})
        self.assertEqual(fb.names("contacts.search", "note"), set())          # not capitalised
        self.assertEqual(FormatBook.from_json(fb.to_json()).names("contacts.search", "query"), names)
        cases = [("mẹ", "nhắn cho mẹ", "Mẹ"), ("Mai", "nhắn sms cho cô giáo Mai nhé", "Cô giáo Mai"),
                 ("Mai", "nhắn cho cô Mai", "Mai"), ("anh Tuấn", "nạp cho anh Tuấn", "Tuấn"),
                 ("Nguyễn Văn Tuấn", "gọi Nguyễn Văn Tuấn", "Nguyễn Văn Tuấn"), ("Lan", "nhắn chị Lan", "Lan"),
                 ("bác ba", "gọi bác Ba", "Bác Ba"), ("Cô giáo Mai", "", "Cô giáo Mai")]
        for v, text, want in cases:
            self.assertEqual(canonical_name(v, names, text), want, (v, text))
        s = self.by_id["train_0004"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        g = Guard(self.cat, tools, s.context, GuardConfig(fix_names=True), fb, text="nhắn tin cho mẹ")
        d = g.check({"type": "read", "name": "contacts.search", "arguments": {"query": "mẹ"}},
                    make_env(self.sp, s.id), g.init_state(s))
        self.assertEqual(d.action["arguments"], {"query": "Mẹ"})

    def test_ask_with_only_optional_params(self):
        def ask(params):
            return {"type": "ask", "name": "sys.ask_missing_info",
                    "arguments": {"tool_name": "banking.transfer", "params": params}}
        s, g, st, env = self._guard("train_0009", ask_needed_only=True)
        self.assertEqual(g.check(ask(["note"]), env, st).kind, "reject")
        self.assertEqual(g.check(ask(["amount", "note"]), env, st).kind, "execute")
        sms = {"type": "ask", "name": "sys.ask_missing_info",
               "arguments": {"tool_name": "sms.send", "params": ["phone_number"]}}
        s, g, st, env = self._guard("train_0001", ask_needed_only=True)
        self.assertEqual(g.check(sms, env, st).kind, "execute")      # one of contact_id / phone_number

    def test_ask_about_a_tool_without_params(self):
        s, g, st, env = self._guard("train_0008")
        d = g.check({"type": "ask", "name": "sys.ask_missing_info",
                     "arguments": {"tool_name": "banking.get_balance", "params": ["balance"]}}, env, st)
        self.assertEqual(d.kind, "reject")
        self.assertIn("không cần tham số", d.message)

    def test_shipped_rules_keep_gold_at_100(self):
        from lifemate.cli import _load_cfg
        cfg = _load_cfg(None)["guard"]
        self.assertTrue(all(cfg[k] for k in ("cache_history_reads", "ask_needed_only", "ask_ids_after_read",
                                             "xref_typed", "fix_names")))
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, guard_cfg=GuardConfig.from_json(cfg))
        res = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, batch_size=4, log_every=0)
        self.assertEqual(Scorer(self.sp, self.cat).score(self.rows(res), self.samples)["score"], 100.0)

    def test_rft_rows_drop_rejected_attempts(self):
        from lifemate.actions import action_text, call_to_action
        from lifemate.agent import Agent, AgentConfig
        from lifemate.llm import ScriptedBackend
        from lifemate.prompt import FEEDBACK_PREFIX, strip_rejected
        s = self.by_id["train_0001"]
        gold = [action_text(call_to_action(c, self.cat)) for c in s.gold_calls]
        bad = '{"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_nobody", "content": "x"}}'
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb)
        ep = build_episode(rc, s)
        Agent(ScriptedBackend({s.id: [bad] + gold + ['{"type": "finish"}']}), AgentConfig()).run([ep])
        self.assertEqual([e["kind"] for e in ep.events][0], "reject")
        self.assertEqual(Scorer(self.sp, self.cat).score_one(s, ep.env.get_tool_calls(), ep.env.get_state_hash())["score"], 1.0)
        msgs = strip_rejected(ep.messages, ep.n_messages_init)
        self.assertEqual(msgs[:ep.n_messages_init], ep.messages[:ep.n_messages_init])
        self.assertFalse(any(str(m["content"]).startswith(FEEDBACK_PREFIX) for m in msgs))
        self.assertEqual([m["content"] for m in msgs[ep.n_messages_init:] if m["role"] == "assistant"], gold)
        self.assertEqual(len(msgs), len(ep.messages) - 2)

    def test_diff_runs_tag_summary(self):
        import contextlib
        import io
        import tempfile
        from lifemate import diffruns
        d = Path(tempfile.mkdtemp())
        a = {"score": 50.0, "per_sample": [{"id": "x", "score": 1.0, "tags": []},
                                           {"id": "y", "score": 0.0, "tags": ["extra_ask:sys.ask_missing_info"]}]}
        b = {"score": 50.0, "per_sample": [{"id": "x", "score": 0.5, "tags": ["read_arg:contacts.search.query"]},
                                           {"id": "y", "score": 1.0, "tags": []}]}
        (d / "a.json").write_text(json.dumps(a), encoding="utf-8")
        (d / "b.json").write_text(json.dumps(b), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diffruns.main(["--a", str(d / "a.json"), "--b", str(d / "b.json")])
        out = buf.getvalue()
        self.assertIn("tags that came with the worse samples: read_arg:contacts.search.query 1", out)
        self.assertIn("tags gone in the better samples: extra_ask:sys.ask_missing_info 1", out)

    def test_rules_learned_from_train_gold(self):
        fb = FormatBook()
        ask = self.cat.ask_format()
        for _ in range(20):
            fb.observe([{"name": "smart_home.get_status", "arguments": {}},
                        {"name": "banking.transfer", "arguments": {"contact_id": "c_me", "amount": 1}},
                        {"name": "banking.transfer", "arguments": {"contact_id": "c_me", "amount": 1}}], ask)
        fb.observe([{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["note"]}}], ask)
        fb2 = FormatBook.from_json(json.loads(json.dumps(fb.to_json())))
        self.assertEqual(fb2.repeatable, {"banking.transfer"})
        self.assertEqual(fb2.asked, {"banking.transfer.note"})
        self.assertEqual(fb2.tool_calls["smart_home.get_status"], 20)
        s = self.by_id["train_0006"]
        tools = self.cat.allowed_tools(s.installed_tool_names()) + ["smart_home.get_status"]
        g = Guard(self.cat, tools, s.context,
                  GuardConfig(drop_unused_params=True, ask_needed_only=True, block_duplicate_writes=True), fb2)
        st, env = g.init_state(s), make_env(self.sp, s.id)
        d = g.check({"type": "read", "name": "smart_home.get_status", "arguments": {"room": "phòng ngủ"}}, env, st)
        self.assertEqual(d.action["arguments"], {})            # gold never filters get_status
        tr = {"type": "write", "name": "banking.transfer", "arguments": {"contact_id": "c_me", "amount": 1}}
        st.writes_done.add(canon_key("banking.transfer", tr["arguments"]))
        self.assertNotEqual(g.check(tr, env, st).kind, "cached")   # gold repeats this write
        asknote = {"type": "ask", "name": "sys.ask_missing_info",
                   "arguments": {"tool_name": "banking.transfer", "params": ["note"]}}
        self.assertEqual(g.check(asknote, env, st).kind, "execute")  # optional, but gold asks for it

    def test_id_seen_under_its_own_key(self):
        s, g, st, env = self._guard("train_0001", xref_typed=True)
        log = {"ok": True, "calls": [{"contact_id": "c_lan", "time": "08:00"}]}
        g.record({"type": "read", "name": "smart_home.get_status", "arguments": {}}, log, st)
        send = {"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "xin chào"}}
        self.assertEqual(g.check(send, env, st).kind, "execute")   # a call log names the contact by contact_id

    def test_diag_runs(self):
        import contextlib
        import io
        import tempfile
        from lifemate import diag
        tr = Path(tempfile.mkdtemp()) / "run.trace.jsonl"
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb)
        run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, trace=tr, log_every=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diag.main(["--data", str(DATA), "--trace", str(tr)])
        out = buf.getvalue()
        self.assertIn("score 100.00", out)
        self.assertIn("gold reuses a history ID: 1 samples", out)
        self.assertIn("x-ref values in gold calls: ref app", out)
        rep = Path(tr).with_suffix(".report.json")
        rep.write_text(json.dumps(Scorer(self.sp, self.cat).score(
            read_jsonl(tr), self.samples)), encoding="utf-8")
        from lifemate import diffruns
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diffruns.main(["--a", str(rep), "--b", str(rep)])
        self.assertIn("score 100.00 -> 100.00 | better 0, worse 0", buf.getvalue())


class TestThoughts(Base):
    def test_actions_keep_the_thought_first(self):
        from lifemate.actions import action_text, parse_action
        a = parse_action('{"thought": "Cần số tiền hoá đơn trước", "type": "read", "name": "bill.list", "arguments": {}}')
        self.assertEqual(a["thought"], "Cần số tiền hoá đơn trước")
        self.assertTrue(action_text(a).startswith('{"thought": "Cần số tiền hoá đơn trước", "type": "read"'))
        self.assertEqual(parse_action('{"thought": "Xong", "type": "finish"}'), {"thought": "Xong", "type": "finish"})
        self.assertNotIn("thought", parse_action('{"thought": " ", "type": "read", "name": "x.y", "arguments": {}}'))

    def test_sft_rows_carry_thoughts(self):
        import tempfile
        from lifemate.prompt import FORMAT_THOUGHT, PromptConfig
        from lifemate.sft_data import OOS_THOUGHT, SFTConfig, build_rows, load_thoughts
        f = Path(tempfile.mkdtemp()) / "th.jsonl"
        s = self.by_id["train_0001"]
        n = len(s.gold_calls)
        rows = [{"id": s.id, "step": k, "thought": f"lý do {k}"} for k in range(n + 1)]
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        th = load_thoughts(f)
        self.assertEqual(th[s.id], [f"lý do {k}" for k in range(n + 1)])
        out, stat = build_rows([s, self.by_id["train_0005"]], self.cat, None, None, PromptConfig(thoughts=True),
                               SFTConfig(n_aug=0, p_asr=0.0, p_confuse=0.0), thoughts=th)
        msgs = out[0]["messages"]
        self.assertIn(FORMAT_THOUGHT, msgs[0]["content"])
        acts = [json.loads(m["content"]) for m in msgs if m["role"] == "assistant"]
        self.assertEqual([a.get("thought") for a in acts], [f"lý do {k}" for k in range(n + 1)])
        self.assertEqual(acts[-1]["type"], "finish")
        oos = [json.loads(m["content"]) for m in out[1]["messages"] if m["role"] == "assistant"]
        self.assertEqual(oos, [{"thought": OOS_THOUGHT, "type": "finish"}])     # train_0005 is out of scope
        self.assertEqual(stat["aug:thought"], 2)

    def test_thoughts_missing_a_step_are_dropped(self):
        from lifemate.prompt import PromptConfig
        from lifemate.sft_data import SFTConfig, build_rows
        s = self.by_id["train_0001"]
        th = {s.id: ["a"] * len(s.gold_calls)}                   # no thought for the closing finish
        out, stat = build_rows([s], self.cat, None, None, PromptConfig(thoughts=True),
                               SFTConfig(n_aug=0, p_asr=0.0, p_confuse=0.0), thoughts=th)
        self.assertEqual(stat.get("thought_incomplete"), 1)
        self.assertNotIn("thought", out[0]["meta"]["aug"])
        oos = self.by_id["train_0005"]
        out, _ = build_rows([oos], self.cat, None, None, PromptConfig(thoughts=True),
                            SFTConfig(n_aug=0, p_asr=0.0, p_confuse=0.0), thoughts={oos.id: ["Chỉ là lời cảm ơn."]})
        self.assertEqual(json.loads(out[0]["messages"][-1]["content"])["thought"], "Chỉ là lời cảm ơn.")

    def test_model_folder_sets_the_prompt_format(self):
        import tempfile
        from lifemate.prompt import (MODEL_PROMPT_FILE, PromptConfig, apply_model_prompt, rows_use_thoughts,
                                     system_rules, write_model_prompt)
        rows = [{"messages": [{"role": "system", "content": system_rules(PromptConfig(thoughts=t))}]}
                for t in (True, True, False)]
        self.assertEqual(rows_use_thoughts(rows), (True, 2))
        self.assertEqual(rows_use_thoughts(rows[2:]), (False, 0))
        run = Path(tempfile.mkdtemp()) / "sft-v3"
        write_model_prompt(run, {"thoughts": True})
        ck = run / "checkpoint-200"
        ck.mkdir()
        cfg = PromptConfig()
        self.assertEqual(apply_model_prompt(cfg, str(ck), "/no/base"), run / MODEL_PROMPT_FILE)
        self.assertTrue(cfg.thoughts)                          # a checkpoint uses its run's file
        cfg = PromptConfig(thoughts=True)
        self.assertIsNone(apply_model_prompt(cfg, None, "/no/base"))
        self.assertTrue(cfg.thoughts)                          # no file: config value stays
        old = Path(tempfile.mkdtemp()) / "sft-v1"
        write_model_prompt(old, {"thoughts": False})
        cfg = PromptConfig(thoughts=True)
        apply_model_prompt(cfg, str(old))
        self.assertFalse(cfg.thoughts)

    def test_type_scoring_is_skipped_with_thoughts(self):
        from lifemate.actions import action_text, call_to_action
        from lifemate.agent import Agent, AgentConfig
        from lifemate.llm import ScriptedBackend
        s = self.by_id["train_0001"]
        script = [action_text({"thought": "x", **call_to_action(c, self.cat)}) for c in s.gold_calls]
        script.append('{"thought": "xong", "type": "finish"}')
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, prompt_cfg=PromptConfig(thoughts=True))
        ep = build_episode(rc, s)
        Agent(ScriptedBackend({s.id: script}), AgentConfig(use_type_scoring=True)).run([ep])   # score() never called
        self.assertEqual(ep.stop_reason, "finish")
        self.assertEqual(len(ep.env.get_tool_calls()), len(s.gold_calls))

    def test_rationale_cli_writes_and_resumes(self):
        import tempfile
        from unittest import mock
        import lifemate.llm as llm_mod
        from lifemate import rationale
        from lifemate.sft_data import load_thoughts

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        class FakeVLLM:
            n_gen = 0

            def __init__(self, **kw):
                self.tok = Tok()

            def render(self, messages):
                return messages[-1]["content"]

            def generate(self, reqs):
                FakeVLLM.n_gen += len(reqs)
                return ["Lý do: dựa vào yêu cầu." for _ in reqs]

        out = Path(tempfile.mkdtemp()) / "th.jsonl"
        args = ["--data", str(DATA), "--model", "x", "--out", str(out), "--limit", "3", "--chunk", "2"]
        with mock.patch.object(llm_mod, "VLLMBackend", FakeVLLM):
            rationale.main(args)
            n = FakeVLLM.n_gen
            rationale.main(args)                                  # second run: nothing left to do
        self.assertEqual(FakeVLLM.n_gen, n)
        th = load_thoughts(out)
        self.assertEqual(len(th), 3)
        self.assertTrue(all(t == "dựa vào yêu cầu." for v in th.values() for t in v))

    def test_agent_runs_with_thoughts(self):
        from lifemate.actions import action_text, call_to_action
        from lifemate.agent import Agent, AgentConfig
        from lifemate.llm import ScriptedBackend
        s = self.by_id["train_0001"]
        script = [action_text({"thought": f"bước {i}", **call_to_action(c, self.cat)}) for i, c in enumerate(s.gold_calls)]
        script.append('{"thought": "Đã nhắn xong", "type": "finish"}')
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb,
                        prompt_cfg=PromptConfig(thoughts=True))
        ep = build_episode(rc, s)
        Agent(ScriptedBackend({s.id: script}), AgentConfig()).run([ep])
        self.assertEqual(Scorer(self.sp, self.cat).score_one(s, ep.env.get_tool_calls(), ep.env.get_state_hash())["score"], 1.0)
        said = [json.loads(m["content"]).get("thought") for m in ep.messages if m["role"] == "assistant"]
        self.assertEqual(said, [f"bước {i}" for i in range(len(s.gold_calls))])
        self.assertEqual([e.get("thought") for e in ep.events], [f"bước {i}" for i in range(len(s.gold_calls))] + ["Đã nhắn xong"])

    def test_teacher_prompts(self):
        from lifemate.rationale import clean, step_list, teacher_messages
        s = self.by_id["train_0006"]                 # history, then a transfer
        acts = step_list(s, self.cat)
        self.assertEqual(acts[-1], {"type": "finish"})
        msgs = teacher_messages(s, PromptConfig(), 1, acts)
        u = msgs[1]["content"]
        self.assertIn("Các bước đã làm:\n1. ", u)
        self.assertIn("BƯỚC TIẾP THEO ĐÚNG: kết thúc", u)
        ask = self.by_id["train_0004"]               # ends with an ask: no finish step
        self.assertEqual(step_list(ask, self.cat)[-1]["type"], "ask")
        self.assertEqual(clean('Lý do: "Cần đọc hoá đơn trước."'), "Cần đọc hoá đơn trước.")
        self.assertIsNone(clean("  "))


class FakeTeacher:
    """Stands in for VLLMBackend in `synth`: new requests on demand, gold steps as the agent.
    Run B (no few-shot block) goes wrong on purpose for the smart_home sample."""
    needs_prompt = True
    calls = 0

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def __init__(self, **kw):
        self.tok = self._Tok()
        self.by_id = {s.id: s for s in load_samples(resolve_split(str(DATA), "train"))}
        self.cat = Catalog.load(resolve_split(str(DATA), "train").catalog)

    def render(self, messages):
        return "\n".join(m["content"] for m in messages)

    def generate(self, reqs):
        from lifemate.actions import action_text, call_to_action
        FakeTeacher.calls += len(reqs)
        out = []
        for r in reqs:
            if r.episode is None:                       # asked for new requests
                out.append('```json\n{"requests": ["yêu cầu mới số một nhé", "yêu cầu mới số hai nhé", '
                           '"yêu cầu mới số một nhé"]}\n```')
                continue
            e = r.episode
            src = self.by_id[e.sample.origin or e.sample.id]
            gold = src.gold_calls or []
            if e.step < len(gold):
                act = call_to_action(gold[e.step], self.cat)
                if src.id == "train_0002" and "# Ví dụ đã làm đúng" not in r.prompt and act["type"] == "write":
                    act = {**act, "arguments": {**act["arguments"], "power": "off" if act["arguments"].get(
                        "power") == "on" else "on"}}
                out.append(action_text({"thought": f"bước {e.step}", **act}))
            else:
                out.append('{"thought": "đã xong", "type": "finish"}')
        return out


class TestSynth(Base):
    def test_parse_requests(self):
        from lifemate.synth import parse_requests
        got = parse_requests('Đây: {"requests": ["Nhắn chị Lan tối nay về muộn", "a", "nhắn chị lan, tối nay về muộn!",'
                             ' "Gọi xe về nhà lúc 6 giờ"]}', 5, "gọi xe về nhà lúc 6 giờ")
        self.assertEqual(got, ["Nhắn chị Lan tối nay về muộn"])   # too short, same after norm, same as original
        self.assertEqual(parse_requests("1. Bật đèn phòng ngủ\n2) Tắt quạt phòng khách\nghi chú", 5),
                         ["Bật đèn phòng ngủ", "Tắt quạt phòng khách"])

    def test_request_prompt_shows_state_and_gold(self):
        from lifemate.synth import request_messages
        s = self.by_id["train_0001"]
        st = make_env(self.sp, s.id).get_state()
        st["_meta"] = {"tools": ["x"]}
        msgs = request_messages(s, self.cat, st, 4, PromptConfig())
        u = msgs[1]["content"]
        self.assertIn("4 yêu cầu", msgs[0]["content"])
        self.assertIn("contacts:", u)
        self.assertNotIn("_meta", u)
        self.assertIn("contacts.search", u.split("# Các bước trợ lý đã làm đúng")[1])

    def test_synth_filters(self):
        from lifemate.synth import same_calls, verdict
        fmap = self.cat.fuzzy_map()
        send = {"name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn nhé"}}
        near = {"name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn"}}
        other = {"name": "sms.send", "arguments": {"contact_id": "c_minh", "content": "tối nay em về muộn nhé"}}
        self.assertIn("content", fmap["sms.send"])
        self.assertTrue(same_calls([send], [near], fmap))          # free text: compared like the scorer does
        self.assertFalse(same_calls([send], [other], fmap))
        self.assertFalse(same_calls([send], [send, send], fmap))
        ev = {"kind": "call", "name": "sms.send", "ok": True, "thought": "x"}
        fin = {"kind": "finish", "thought": "xong"}
        run = {"stop_reason": "finish", "events": [ev, fin], "tool_calls": [send], "state_hash": "h"}
        self.assertEqual(verdict(run, dict(run), False, fmap), "ok")
        self.assertEqual(verdict(run, {**run, "tool_calls": [other]}, False, fmap), "disagree")
        self.assertEqual(verdict(run, None, False, fmap), "no_run_b")
        self.assertEqual(verdict(run, None, False, fmap, need_b=False), "ok")
        noop = {**run, "events": [{**ev, "noop": True}, fin]}
        self.assertEqual(verdict(noop, dict(noop), False, fmap), "noop")      # unknown ID, short balance...
        empty = {"stop_reason": "finish", "events": [fin], "tool_calls": [], "state_hash": "h"}
        self.assertEqual(verdict(empty, dict(empty), False, fmap), "no_call")
        self.assertEqual(verdict(empty, dict(empty), True, fmap), "ok")      # the source was out of scope too

    def test_vllm_skips_prompts_that_do_not_fit(self):
        import types
        from unittest import mock
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **kw: kw
        from lifemate.llm import GenRequest, VLLMBackend
        b = object.__new__(VLLMBackend)
        b.max_model_len, b.lora_req = 100, None
        b.tok = types.SimpleNamespace(encode=lambda t, add_special_tokens=False: t.split())
        seen = {}

        def gen(prompts, params, **kw):
            seen["max"] = [p["max_tokens"] for p in params]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"o{i}")]) for i in range(len(prompts))]

        b.llm = types.SimpleNamespace(generate=gen)
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            got = b.generate([GenRequest(None, "a " * 50, "", 0.0, 512), GenRequest(None, "a " * 95, "", 0.0, 512)])
        self.assertEqual(got, ["o0", ""])
        self.assertEqual(seen["max"], [50])

    def test_synth_end_to_end(self):
        import tempfile
        from unittest import mock
        import lifemate.llm as llm_mod
        from lifemate import synth
        from lifemate.sft_data import SFTConfig, build_rows, load_thoughts
        d = Path(tempfile.mkdtemp())
        out, th = d / "synth.jsonl", d / "synth_th.jsonl"
        args = ["--data", str(DATA), "--model", "x", "--k", "2", "--out", str(out), "--thoughts-out", str(th),
                "--limit", "5", "--chunk", "2", "--batch", "3"]
        with mock.patch.object(llm_mod, "VLLMBackend", FakeTeacher):
            synth.main(args)
            n = FakeTeacher.calls
            synth.main(args)                              # resumes: nothing left
        self.assertEqual(FakeTeacher.calls, n)
        rows = read_jsonl(out)
        by_src = {}
        for r in rows:
            by_src.setdefault(r["origin"], []).append(r)
        self.assertEqual(sorted(by_src), ["train_0001", "train_0003", "train_0004", "train_0005"])  # 0002: B differs
        self.assertEqual(len(by_src["train_0001"]), 2)    # the repeated request is dropped
        r = by_src["train_0001"][0]
        self.assertEqual(r["env_id"], "train_0001")
        self.assertEqual(r["label"]["tool_calls"], self.by_id["train_0001"].gold_calls)
        self.assertTrue(all(x["ok"] for x in r["label"]["tool_results"]))
        self.assertEqual(by_src["train_0005"][0]["label"]["tool_calls"], [])     # out of scope stays so
        thoughts = load_thoughts(th)
        self.assertEqual(thoughts[r["id"]], ["bước 0", "bước 1", "đã xong"])
        self.assertEqual(thoughts[by_src["train_0004"][0]["id"]], ["bước 0", "bước 1"])   # ends with the ask
        pool = FewShotPool.from_samples(self.samples, self.cat)
        syn = [Sample(x) for x in rows]
        all_th = {**thoughts, **{s.id: [f"t{i}" for i in range(len(s.gold_calls) + 1)] for s in self.samples}}
        sft, stat = build_rows(syn, self.cat, None, pool, PromptConfig(thoughts=True),
                               SFTConfig(n_aug=1, p_confuse=0.0), thoughts=all_th)
        self.assertEqual(stat["aug:synth"], len(syn))     # one row each (n_aug_synth 0)
        one = next(x for x in sft if x["meta"]["id"] == r["id"])
        self.assertNotIn(self.by_id["train_0001"].audio_text, one["messages"][0]["content"])   # no source example
        self.assertIn("yêu cầu mới số", one["messages"][1]["content"])

    def test_synth_check_gate(self):
        import tempfile
        from unittest import mock
        import lifemate.llm as llm_mod
        from lifemate import synth
        d = Path(tempfile.mkdtemp())
        ids = d / "val.txt"
        ids.write_text("train_0001\ntrain_0002\ntrain_0004\n", encoding="utf-8")
        rep = d / "check.json"
        args = ["--data", str(DATA), "--model", "x", "--check-ids", str(ids), "--check-only", "--report", str(rep)]
        with mock.patch.object(llm_mod, "VLLMBackend", FakeTeacher):
            synth.main(args)
            got = json.loads(rep.read_text(encoding="utf-8"))
            self.assertEqual((got["n"], got["kept"], got["kept_score"], got["passed"]), (3, 2, 100.0, True))
            self.assertEqual(got["why"].get("disagree"), 1)
            n = FakeTeacher.calls
            with self.assertRaises(SystemExit) as cm:          # stored result, stricter threshold: no new runs
                synth.main(args + ["--min-kept-score", "101"])
            self.assertEqual(cm.exception.code, 3)
            self.assertEqual(FakeTeacher.calls, n)
            rep.unlink()
            with self.assertRaises(SystemExit) as cm:
                synth.main(args + ["--min-kept-score", "101"])
            self.assertEqual(cm.exception.code, 3)
            self.assertGreater(FakeTeacher.calls, n)


class TestASRContext(Base):
    HIST = [
        {"role": "user", "audio": "u1.wav", "audio_text": "tìm số của chị Hoa"},
        {"role": "assistant", "text": "Dạ, chị Hoa Bích là bạn của chị ạ.",
         "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Hoa"}}],
         "tool_results": [{"ok": True, "contacts": [
             {"id": "c_07", "name": "Hoa Bích", "relation": "bạn", "phone": "0901234567"},
             {"id": "c_08", "name": "Peter Lindqvist", "relation": "đối tác", "status": "active"}]}]},
        {"role": "user", "audio": "u2.wav", "audio_text": "nhắn cho anh ấy"},
        {"role": "assistant", "text": "Anh muốn nhắn Peter qua Zalo hay SMS?",
         "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["x"]}}],
         "tool_results": [{"ok": True, "waiting_for_user": True}]},
    ]

    def sample(self):
        return Sample({"id": "x", "audio": "x.wav", "history": self.HIST})

    def test_prompt_has_reply_and_names_not_ids(self):
        from lifemate.asr.context import MAX_PROMPT_CHARS, prompt_for
        s = self.sample()
        p = prompt_for(s)
        self.assertTrue(p.endswith("Anh muốn nhắn Peter qua Zalo hay SMS?"), p)
        for want in ("Peter Lindqvist", "Hoa Bích", "đối tác"):
            self.assertIn(want, p)
        for bad in ("c_07", "0901234567", "active", "contacts.search", "sms.send"):
            self.assertNotIn(bad, p)
        self.assertLessEqual(len(p), MAX_PROMPT_CHARS)
        self.assertEqual(prompt_for(s, 0), "")                  # nothing came before the first user turn
        self.assertTrue(prompt_for(s, 1).endswith("bạn của chị ạ."))
        self.assertNotIn("Zalo", prompt_for(s, 1))               # only turns before that user turn
        self.assertLessEqual(len(prompt_for(s, max_chars=60)), 60)

    def test_clip_and_utterance_prompts(self):
        from lifemate.asr.context import clip_prompt_ids
        from lifemate.asr.transcribe import utterance_prompts
        self.assertEqual(clip_prompt_ids([9, 1, 2, 3, 4, 5], 4), [9, 3, 4, 5])
        self.assertEqual(clip_prompt_ids([9, 1], 4), [9, 1])
        s = self.sample()
        self.assertEqual(utterance_prompts(s, False), [None, None, None])
        ps = utterance_prompts(s, True)
        self.assertTrue(ps[0] and ps[1] is None and ps[2] is None)

    def test_decoder_pairs_mask_prompt(self):
        from lifemate.asr.finetune_whisper import decoder_lists
        sot, pad, prev = 100, 0, 99
        lab = [sot, 1, 2, 3, 4, 5]                               # <|sot|> <|vi|> ... <|eot|>=5
        dec, tgt = decoder_lists([lab], [[]], sot, pad)
        self.assertEqual((dec[0], tgt[0]), ([sot, 1, 2, 3, 4], [1, 2, 3, 4, 5]))   # as before prompts
        dec, tgt = decoder_lists([lab, lab], [[prev, 7, 8], []], sot, pad)
        self.assertEqual(dec[0], [prev, 7, 8, sot, 1, 2, 3, 4])
        self.assertEqual(tgt[0], [-100, -100, -100, 1, 2, 3, 4, 5])
        self.assertEqual(dec[1], [sot, 1, 2, 3, 4, pad, pad, pad])
        self.assertEqual(tgt[1], [1, 2, 3, 4, 5, -100, -100, -100])
        dec, tgt = decoder_lists([lab], [[prev, 7, 8, 9]], sot, pad, max_len=8)    # 4 + 1 + 5 = 10 > 8
        self.assertEqual(dec[0], [prev, 9, sot, 1, 2, 3, 4])
        self.assertEqual(tgt[0], [-100, -100, 1, 2, 3, 4, 5])

    def test_transcribe_final_only_with_prompts(self):
        import tempfile
        from lifemate.asr.transcribe import transcribe_samples

        class Rec:
            def __init__(self):
                self.calls = []

            def transcribe(self, paths, batch_size=16, prompts=None):
                self.calls.append((list(paths), prompts))
                return [f"t{i}" for i in range(len(paths))]

        s = self.by_id["train_0006"]                             # one history turn
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.jsonl"
            old.write_text(json.dumps({"id": s.id, "final": "a", "history": ["cũ"]}) + "\n", encoding="utf-8")
            eng = Rec()
            transcribe_samples(eng, self.sp, [s], Path(d) / "new.jsonl", prompt_final=True, history_from=old,
                               log_every=0)
            row = read_jsonl(Path(d) / "new.jsonl")[0]
        self.assertEqual(row, {"id": s.id, "final": "t0", "history": ["cũ"]})
        (paths, prompts), = eng.calls
        self.assertEqual(len(paths), 1)
        self.assertIn("mẹ", prompts[0])

    def test_whisper_strip_prompt(self):
        from lifemate.asr.engines import Whisper
        w = object.__new__(Whisper)
        w.sot = 100
        self.assertEqual(w._strip([99, 7, 8, 100, 1, 2, 5]), [1, 2, 5])
        self.assertEqual(w._strip([1, 2, 5]), [1, 2, 5])

    def test_predict_reads_asr_settings(self):
        import tempfile
        from unittest import mock
        from lifemate import predict as P
        from lifemate.asr import engines
        from lifemate.asr.context import write_settings
        got = {}

        def fake(kind, model, lora=None, device="cuda", num_beams=1):
            got["beams"] = num_beams
            return object()

        with tempfile.TemporaryDirectory() as d, mock.patch.object(engines, "make_engine", fake):
            write_settings(d, prompt_final=True, beams=4)
            pr = object.__new__(P.Predictor)
            for kw, want in ((dict(), (4, True)), (dict(asr_beams=2), (2, True)),
                             (dict(asr_prompt="off"), (4, False))):
                pr.cfg = P.PredictConfig(asr="whisper", asr_model=d, **kw)
                pr._load_asr()
                self.assertEqual((got["beams"], pr.asr_prompt), want, kw)
            pr.cfg = P.PredictConfig(asr="whisper", asr_model=str(Path(d) / "none"))
            pr._load_asr()
            self.assertEqual((got["beams"], pr.asr_prompt), (1, False))


class FakeLM:
    """Scores a text by how many 'good' words it holds; .tok counts whitespace tokens."""
    GOOD = {"kém", "sáng", "nạp"}

    class Tok:
        @staticmethod
        def encode(t, add_special_tokens=False):
            return t.split()

    tok = Tok()

    def score(self, reqs):
        return [[3.0 * sum(w in self.GOOD for w in c.split()) for c in r.candidates] for r in reqs]


class TestFusion(Base):
    def test_regions_and_apply(self):
        from lifemate.asr.fuse import apply, regions, variants
        toks, sp = regions("Nhắc tôi tám giờ trễ mười lăm tối mai nha.", "nhắc tôi tám giờ kém mười lăm tối mai nhé")
        self.assertEqual([(" ".join(toks[r.start:r.end]), r.words) for r in sp], [("trễ", ["kém"])])  # nha/nhé same
        self.assertEqual(apply(toks, sp, [True]), "Nhắc tôi tám giờ kém mười lăm tối mai nha.")
        self.assertEqual(variants(toks, sp)[0], "Nhắc tôi tám giờ trễ mười lăm tối mai nha.")
        toks, sp = regions("nạp năm mươi nghìn", "nạp 50 ngàn")                 # format only: no span
        self.assertEqual(sp, [])
        toks, sp = regions("a b c", "a x b c")                                   # insertion
        self.assertEqual(apply(toks, sp, [True]), "a x b c")
        self.assertEqual(regions("hoàn toàn khác nhau", "một hai ba bốn năm")[1], [])  # too different
        self.assertEqual(regions("", "abc")[1], [])

    def test_choose_tune_and_fuser(self):
        from lifemate.asr import fuse as F
        self.assertEqual(F.choose([0.0, 5.0, -1.0], [3, 3, 3], 1.0, 0.0), [True, False])
        self.assertEqual(F.choose([0.0, 0.5], [3, 5], 1.0, 0.5), [True])        # longer variant, beta > 0
        s = self.by_id["train_0006"]
        out = F.Fuser(FakeLM(), delta=1.0).fuse_many([(s, "đạp hai trăm nghìn tháng mai", "nạp hai trăm nghìn sáng mai"),
                                                       (s, "giống nhau", "giống nhau")])
        self.assertEqual(out, ["nạp hai trăm nghìn sáng mai", "giống nhau"])
        self.assertEqual(F.Fuser(FakeLM(), delta=10.0).fuse_many([(s, "đạp x", "nạp x")]), ["đạp x"])
        self.assertTrue(F.lm_prefix(s).startswith("Trợ lý: "))
        # tune on the mock split: the secondary is the gold text, so fusing toward it cannot be worse
        prim, sec = Transcripts(), Transcripts()
        for x in self.samples:
            g = x.transcripts()[0] or ""
            w = g.split()
            prim.rows[x.id] = {"id": x.id, "final": " ".join(["lỗi"] + w[1:]) if len(w) > 2 else g, "history": []}
            sec.rows[x.id] = {"id": x.id, "final": g, "history": []}

        class AlwaysSecond(FakeLM):
            def score(self, reqs):
                return [[0.0] + [5.0] * (len(r.candidates) - 1) for r in reqs]

        scores = F.score_samples(F.Fuser(AlwaysSecond()), self.samples, prim, sec)
        params, table = F.tune(self.samples, prim, sec, scores)
        self.assertLess(params["delta"], 5.0)
        self.assertEqual(params["wer_final"], 0.0)
        fused = F.fused_finals(self.samples, prim, sec, scores, params["delta"], params["beta"])
        self.assertEqual(F.final_wer(self.samples, fused), 0.0)

    def test_idle_finish_retry(self):
        from lifemate.agent import idle_finish
        self.assertTrue(idle_finish("Người dùng yêu cầu đặt mua dầu ô liu, đã đủ thông tin nên kết thúc hành động."))
        self.assertFalse(idle_finish("Yêu cầu vi phạm bảo mật, không thể thực hiện."))
        self.assertFalse(idle_finish("Mai không mưa nên không cần hủy lịch."))
        self.assertFalse(idle_finish(None))
        s = self.by_id["train_0001"]
        finish = '{"thought": "Người dùng muốn nhắn cho Lan, đã đủ thông tin.", "type": "finish"}'
        script = {s.id: [finish,
                         '{"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}',
                         '{"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", '
                         '"content": "tối nay em về muộn"}}', '{"type": "finish"}']}
        for on, want in ((False, []), (True, ["contacts.search", "sms.send"])):
            rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb,
                            agent_cfg=AgentConfig(retry_idle_finish=on))
            res = run_split(rc, ScriptedBackend({k: list(v) for k, v in script.items()}), [s], log_every=0)[0]
            self.assertEqual([c["name"] for c in res["tool_calls"]], want)

    def test_predict_fusion_path(self):
        import tempfile
        from unittest import mock
        from lifemate import predict as P
        from lifemate.asr import engines

        class Eng:
            def __init__(self, text):
                self.text = text

            def transcribe(self, paths, batch_size=16, prompts=None):
                return [self.text for _ in paths]

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fuse.json").write_text(json.dumps({"delta": 1.0, "beta": 0.0}))
            (Path(d) / "asr2").mkdir()
            pr = object.__new__(P.Predictor)
            pr.cfg = P.PredictConfig(asr="whisper", asr_model=d, assets=d, asr2_model=str(Path(d) / "asr2"))
            pr.llm = FakeLM()
            with mock.patch.object(engines, "make_engine", lambda kind, *a, **k: Eng(
                    "nạp tiền sáng mai" if kind == "auto" else "đạp tiền tháng mai")):
                pr.asr = pr._load_asr()
            self.assertIsNotNone(pr.asr2)
            pr.sp = self.sp
            s = self.by_id["train_0001"]
            final, hist = pr.transcribe(s)
            self.assertEqual(final, "nạp tiền sáng mai")
            pr.cfg = P.PredictConfig(asr="whisper", asr_model=d, assets=d, asr2_model=str(Path(d) / "nope"))
            with mock.patch.object(engines, "make_engine", lambda *a, **k: Eng("x")):
                pr._load_asr()
            self.assertIsNone(pr.asr2)                                   # no second ASR: fusion off


class TestRecovery(Base):
    """Empty searches, calls that change nothing, asks that come too early, repeated reads, votes, date hints."""
    SEARCH = '{"type": "read", "name": "contacts.search", "arguments": {"query": "%s"}}'
    SEND = '{"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn"}}'
    FINISH = '{"type": "finish"}'

    def run_one(self, sid, script, guard=None, **agent):
        from lifemate.agent import Agent
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb,
                        guard_cfg=guard or GuardConfig(), agent_cfg=AgentConfig(**agent))
        ep = build_episode(rc, self.by_id[sid])
        Agent(ScriptedBackend({sid: list(script)}), rc.agent_cfg).run([ep])
        return ep

    def test_outcome(self):
        from lifemate.recovery import outcome
        self.assertEqual(outcome("read", {"query": "x"}, {"ok": True, "results": []}), "empty")
        self.assertIsNone(outcome("read", {}, {"ok": True, "results": []}))            # nothing stored: a fact
        self.assertIsNone(outcome("read", {"query": "x"}, {"ok": True, "results": [1]}))
        self.assertIsNone(outcome("read", {"n": "1"}, {"ok": True, "won": False, "prizes": []}))   # an answer
        self.assertEqual(outcome("write", {"id": "x"}, {"ok": True, "done": False, "message": "m"}), "noop")
        self.assertEqual(outcome("read", {"symbol": "X"}, {"ok": True, "done": False}), "noop")
        self.assertIsNone(outcome("write", {"id": "x"}, {"ok": False, "error": "schema"}))

    def test_hint_names_what_to_change(self):
        from lifemate.recovery import hint
        raw = {"apps": [{"id": "trips", "tools": [
            {"name": "trips.search", "kind": "read", "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"},
                "type": {"type": "string", "enum": ["train", "bus"]}}, "required": ["origin", "destination", "date"]}},
            {"name": "trips.my_bookings", "kind": "read", "parameters": {"type": "object", "properties": {}}}]}]}
        cat = Catalog(raw)
        search, readers = cat.tools["trips.search"], list(cat.tools.values())
        h = hint("empty", search, {"origin": "HAN", "destination": "DAD", "date": "2026-10-03", "type": "train"},
                 {"user_profile": {"city": "Đà Nẵng"}}, readers)
        for part in ("(type)", "origin, destination", "không tự đổi thành mã", "Đà Nẵng", "date:"):
            self.assertIn(part, h)
        self.assertNotIn("type:", h)                                  # an enum is not a keyword
        self.assertNotIn("my_bookings", h)                            # lists something else
        search.params["origin"].raw["description"] = "Mã IATA hoặc tên thành phố"
        self.assertNotIn("không tự đổi thành mã", hint("empty", search, {"origin": "HAN"}, None, readers))
        n = hint("noop", search, {}, None, readers, "không tìm thấy mã")
        self.assertIn("không tìm thấy mã", n)
        self.assertIn("trips.my_bookings", n)

    def test_empty_search_gets_a_second_try(self):
        script = [self.SEARCH % "Lann", self.FINISH, self.SEARCH % "Lan", self.SEND, self.FINISH]
        off = self.run_one("train_0001", script)
        self.assertEqual([c["name"] for c in off.env.get_tool_calls()], ["contacts.search"])
        self.assertTrue(off.events[0].get("empty"))
        self.assertNotIn("Lưu ý", off.messages[-1]["content"])
        on = self.run_one("train_0001", script, recover=True)
        self.assertEqual([c["name"] for c in on.env.get_tool_calls()], ["contacts.search", "contacts.search", "sms.send"])
        self.assertEqual(on.stop_reason, "finish")
        notes = [m["content"] for m in on.messages if m["role"] == "user" and "Lưu ý: không có kết quả" in m["content"]]
        self.assertEqual(len(notes), 1)
        self.assertIn("query", notes[0])
        self.assertEqual([e["kind"] for e in on.events], ["call", "reject", "call", "call", "finish"])

    def test_second_empty_search_may_stop(self):
        # tried again by itself and still nothing: the finish stands, no feedback
        ep = self.run_one("train_0001", [self.SEARCH % "Lann", self.SEARCH % "Lanh", self.FINISH], recover=True)
        self.assertEqual(len(ep.env.get_tool_calls()), 2)
        self.assertEqual([e["kind"] for e in ep.events], ["call", "call", "finish"])
        # sent back once only: a model that insists is let go
        ep = self.run_one("train_0001", [self.SEARCH % "Lann", self.FINISH, self.FINISH], recover=True)
        self.assertEqual(len(ep.env.get_tool_calls()), 1)
        self.assertEqual([e["kind"] for e in ep.events], ["call", "reject", "finish"])

    def test_ask_right_after_empty_search_is_sent_back(self):
        ask = '{"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}'
        ep = self.run_one("train_0001", [self.SEARCH % "Lann", ask, self.SEARCH % "Lan", self.SEND, self.FINISH],
                          recover=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "contacts.search", "sms.send"])

    def test_noop_write(self):
        bad = '{"type": "write", "name": "smart_home.light_set", "arguments": {"device_id": "light_x", "power": "off"}}'
        status = '{"type": "read", "name": "smart_home.get_status", "arguments": {}}'
        good = '{"type": "write", "name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off"}}'
        g = GuardConfig(check_xref=False, block_failed_repeats=True)
        ep = self.run_one("train_0002", [bad, self.FINISH, bad, status, good, self.FINISH], guard=g, recover=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()],
                         ["smart_home.light_set", "smart_home.get_status", "smart_home.light_set"])
        self.assertTrue(ep.events[0].get("noop"))
        errors = [e["error"] for e in ep.events if e["kind"] == "reject"]
        self.assertEqual(len(errors), 2)                               # the finish, then the same write again
        self.assertIn("chưa có tác dụng", errors[0])
        self.assertIn("không có tác dụng", errors[1])
        self.assertTrue(any("Lưu ý: lệnh chưa có tác dụng" in m["content"] for m in ep.messages))
        # looking the data up is not yet the retry: a finish right after it is still sent back
        ep = self.run_one("train_0002", [bad, status, self.FINISH, good, self.FINISH], guard=g, recover=True)
        self.assertEqual([e["kind"] for e in ep.events], ["call", "call", "reject", "call", "finish"])
        self.assertEqual(len(ep.env.get_tool_calls()), 3)

    def test_deferred_ask_is_sent_last(self):
        ask = '{"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}'
        ev = ('{"type": "write", "name": "calendar.create_event", "arguments": {"title": "họp team", '
              '"start_time": "2026-10-21T08:00:00+07:00"}}')
        now = self.run_one("train_0005", [ask, ev, self.FINISH])
        self.assertEqual([c["name"] for c in now.env.get_tool_calls()], ["sys.ask_missing_info"])
        ep = self.run_one("train_0005", [ask, ev, self.FINISH], defer_ask=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["calendar.create_event", "sys.ask_missing_info"])
        self.assertEqual(ep.stop_reason, "ask")
        self.assertTrue(any("gửi cho người dùng sau cùng" in m["content"] for m in ep.messages))
        # the tool the question is about does not run, a second question is dropped, nothing else to do = just the ask
        send = '{"type": "write", "name": "sms.send", "arguments": {"phone_number": "0901111222", "content": "a"}}'
        for script in ([ask, send, ev], [ask, ask.replace("contact_id", "content"), ev], [ask, self.FINISH]):
            ep = self.run_one("train_0005", script, defer_ask=True)
            self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["sys.ask_missing_info"], script)
            self.assertEqual(ep.stop_reason, "ask")
        # the step limit also sends it
        ep = self.run_one("train_0005", [ask], defer_ask=True, max_steps=1)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["sys.ask_missing_info"])

    def test_no_text_ask(self):
        s = self.by_id["train_0004"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        ask = {"type": "ask", "name": "sys.ask_missing_info",
               "arguments": {"tool_name": "sms.send", "params": ["content"]}}
        for on in (False, True):
            g = Guard(self.cat, tools, s.context, GuardConfig(no_text_ask=on), self.fb)
            st = g.init_state(s)
            self.assertEqual(g.check(ask, make_env(self.sp, s.id), st).kind, "reject" if on else "execute")
            self.assertEqual(g.check(ask, make_env(self.sp, s.id), st).kind, "execute")    # sent back once only
        g = Guard(self.cat, tools, s.context, GuardConfig(no_text_ask=True), self.fb)
        both = {**ask, "arguments": {"tool_name": "sms.send", "params": ["content", "phone_number"]}}
        self.assertEqual(g.check(both, make_env(self.sp, s.id), g.init_state(s)).kind, "execute")

    def test_no_ask_when_one_hit(self):
        from lifemate.guard import note_hits
        s = self.by_id["train_0004"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        ask = {"type": "ask", "name": "sys.ask_missing_info",
               "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}
        g = Guard(self.cat, tools, s.context, GuardConfig(no_ask_single_hit=True), self.fb)
        for n, want in ((1, "reject"), (2, "execute"), (0, "execute")):
            st = g.init_state(s)
            st.read_apps.add("contacts")
            note_hits(self.cat, st.hits, "contacts.search", {"ok": True, "contacts": [{"id": f"c{i}"} for i in range(n)]})
            self.assertEqual(st.hits, {"contacts.contacts": n})
            self.assertEqual(g.check(ask, make_env(self.sp, s.id), st).kind, want, n)
            self.assertEqual(g.check(ask, make_env(self.sp, s.id), st).kind, "execute")    # the model insists: let it
        st = g.init_state(s)
        note_hits(self.cat, st.hits, "sms.send", {"ok": True, "contacts": [1]})        # a write result says nothing
        self.assertEqual(st.hits, {})

    def test_optional_text_dropped_on_unseen_tool(self):
        s = self.by_id["train_0009"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        call = {"type": "write", "name": "banking.transfer",
                "arguments": {"contact_id": "c_hang", "amount": 1200000, "note": "tiền ăn"}}
        ev = {"type": "write", "name": "calendar.create_event",
              "arguments": {"title": "họp", "start_time": "2026-10-21T08:00:00+07:00"}}
        for fb, want in ((FormatBook(), False), (self.fb, True)):        # unseen tool / tool seen in train
            g = Guard(self.cat, tools, s.context, GuardConfig(drop_optional_text_unseen=True), fb)
            st = g.init_state(s)
            st.seen.add("c_hang")
            d = g.check(call, make_env(self.sp, s.id), st)
            self.assertEqual("note" in d.action["arguments"], want)
        s3 = self.by_id["train_0003"]
        g = Guard(self.cat, self.cat.allowed_tools(s3.installed_tool_names()), s3.context,
                  GuardConfig(drop_optional_text_unseen=True), FormatBook())
        d = g.check(ev, make_env(self.sp, s3.id), g.init_state(s3))
        self.assertIn("title", d.action["arguments"])                   # required text stays

    def test_repeated_read_says_so(self):
        status = '{"type": "read", "name": "smart_home.get_status", "arguments": {}}'
        for on in (False, True):
            ep = self.run_one("train_0002", [status, status, self.FINISH], notice_repeat=on)
            self.assertEqual(len(ep.env.get_tool_calls()), 1)
            self.assertEqual("đã gọi với đúng các tham số này" in ep.messages[-1]["content"], on)

    def test_vote(self):
        a, b = self.SEARCH % "Lan", self.SEARCH % "Minh"
        ep = self.run_one("train_0001", [a, b, b, self.FINISH, self.FINISH, a], vote_k=3)
        self.assertEqual(ep.env.get_tool_calls(), [{"name": "contacts.search", "arguments": {"query": "Minh"}}])
        self.assertEqual(ep.events[0]["vote"], {"k": 3, "top": 2, "greedy": False,
                                                "split": [["read", 1, 1], ["read", 2, 1]]})
        # a tie goes to the greedy candidate; free text does not split the vote
        ep = self.run_one("train_0001", [a, b, "not json", self.FINISH, self.FINISH, self.FINISH], vote_k=3)
        self.assertEqual(ep.env.get_tool_calls()[0]["arguments"], {"query": "Lan"})
        s1 = self.SEND
        s2 = s1.replace("tối nay em về muộn", "em về muộn nha")
        ep = self.run_one("train_0001", [a, a, a, b, s1, s2, self.FINISH, self.FINISH, self.FINISH], vote_k=3)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "sms.send"])
        self.assertEqual(ep.env.get_tool_calls()[1]["arguments"]["content"], "tối nay em về muộn")
        # two candidates the guard would refuse (an ID never read) lose against one it would run
        bad = s1.replace("c_lan", "c_999")
        ep = self.run_one("train_0001", [a, a, a, bad, bad, s1, self.FINISH, self.FINISH, self.FINISH], vote_k=3)
        self.assertEqual(ep.env.get_tool_calls()[1]["arguments"]["contact_id"], "c_lan")
        self.assertFalse(any(e["kind"] == "reject" for e in ep.events))
        # the look at the candidates does not use up a one-time warning of the guard
        ask = '{"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}'
        ep = self.run_one("train_0001", [ask, ask, ask, self.FINISH, self.FINISH, self.FINISH],
                          guard=GuardConfig(no_text_ask=True), vote_k=3)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "finish"])

    def test_finish_that_says_information_is_missing(self):
        lost = '{"thought": "Người dùng chưa cung cấp nội dung cần nhắn.", "type": "finish"}'
        script = [self.SEARCH % "Lan", lost, self.SEND, self.FINISH]
        ep = self.run_one("train_0001", script)
        self.assertEqual(len(ep.env.get_tool_calls()), 1)
        ep = self.run_one("train_0001", script, recover_lost=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "sms.send"])
        self.assertIn("Thiếu thông tin", [e for e in ep.events if e["kind"] == "reject"][0]["error"])
        # before any call it may be a request no app can serve; other thoughts are not touched
        ep = self.run_one("train_0001", [lost, self.SEND], recover_lost=True)
        self.assertEqual(ep.env.get_tool_calls(), [])
        done = '{"thought": "Đã nhắn xong, không còn gì cần làm.", "type": "finish"}'
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", done, self.SEND], recover_lost=True)
        self.assertEqual(len(ep.env.get_tool_calls()), 1)

    def test_extra_rules_only_when_asked(self):
        from lifemate.prompt import SYSTEM_RULES, system_rules
        self.assertEqual(system_rules(PromptConfig()), SYSTEM_RULES)       # what the model was trained on
        on = system_rules(PromptConfig(extra_rules=True, thoughts=True))
        self.assertIn("10. Thứ người dùng nhắc tới", on)
        self.assertIn('"thought"', on)

    def test_trace_keeps_results(self):
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", self.FINISH], trace_results=40)
        self.assertTrue(ep.events[0]["res"].startswith('{"ok": true'))
        self.assertLessEqual(len(ep.events[0]["res"]), 40)
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", self.FINISH])
        self.assertNotIn("res", ep.events[0])

    def test_date_hints(self):
        from lifemate.datehints import hints, resolve
        from lifemate.prompt import render_request
        ctx = {"current_time": "2026-10-20T19:30:00+07:00"}          # a Tuesday
        cases = [("đặt vé thứ bảy tuần sau đi Đà Nẵng", "2026-10-31"), ("chủ nhật tuần này về quê", "2026-10-25"),
                 ("thứ sáu này đi ăn", "2026-10-23"), ("thứ ba tới họp", "2026-10-27"),
                 ("thứ hai tuần sau nữa", "2026-11-02"), ("đi khám ngày hai mươi ba tháng mười một", "2026-11-23"),
                 ("mùng 7 tháng 12 về quê", "2026-12-07"), ("ngày 5/1 nhé", "2027-01-05"),
                 ("ngày hai mươi lăm tháng này", "2026-10-25"), ("mùng 2 tháng sau", "2026-11-02")]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual([d.isoformat() for _, d in resolve(text, ctx)], [want])
        self.assertEqual([d.isoformat() for _, d in resolve("họp lúc 9 giờ thứ năm 12 tháng 11", ctx)], ["2026-11-12"])
        # one reading only: a bare weekday or day, a rank, a span of time, a score get no date
        for text in ("thứ ba họp nhé", "ngày 5 đi", "cái thứ hai này đẹp hơn cái kia", "hai tuần sau", "31 tháng 2",
                     "lúc 8/10 điểm", "tỉ số 2/1", "trong ba tháng hai lần", "hai tháng 3 triệu"):
            with self.subTest(text=text):
                self.assertEqual(resolve(text, ctx), [])
        self.assertEqual(hints("thứ bảy tuần sau", ctx), ["thứ bảy tuần sau = 2026-10-31 (thứ bảy)"])
        on, off = PromptConfig(date_hints=True), PromptConfig()
        self.assertIn("Ngày: thứ bảy tuần sau = 2026-10-31", render_request("đặt vé thứ bảy tuần sau", on, ctx))
        self.assertNotIn("Ngày:", render_request("đặt vé thứ bảy tuần sau", off, ctx))
        self.assertNotIn("Ngày:", render_request("đặt vé đi Đà Nẵng", on, ctx))


class TestActiveRecovery(Base):
    """v34: the agent searches again itself, the model rewrites a search, votes that cannot stop a sample,
    asks that name a parameter the tool does not have."""
    SEARCH, SEND, FINISH, run_one = TestRecovery.SEARCH, TestRecovery.SEND, TestRecovery.FINISH, TestRecovery.run_one
    LIST = '{"type": "read", "name": "calendar.list_events", "arguments": {"date": "%s"}}'
    NEW = ('{"type": "write", "name": "calendar.create_event", "arguments": {"title": "họp", '
           '"start_time": "2026-10-21T08:00:00+07:00"}}')
    ASK = '{"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["%s"]}}'

    def test_relaxed_calls(self):
        from lifemate.recovery import relaxed
        search, cal = self.cat.tools["contacts.search"], self.cat.tools["calendar.list_events"]
        self.assertEqual(relaxed(search, {"query": "chị Hằng"}), [({"query": "Hằng"}, "title"), ({}, "all")])
        self.assertEqual(relaxed(search, {"query": "cô giáo Mai"})[0], ({"query": "Mai"}, "title"))
        self.assertEqual(relaxed(search, {"query": "Mẹ"}), [({}, "all")])       # a title alone is the name
        self.assertEqual(relaxed(search, {"query": "Lan Anh"}), [({}, "all")])
        self.assertEqual(relaxed(search, {}), [])                               # the list itself was empty
        self.assertEqual(relaxed(cal, {"date": "2026-10-21"}), [])              # a required filter: no list without it

    def test_agent_searches_without_the_title(self):
        send = self.SEND.replace("c_lan", "c_hang")
        script = [self.SEARCH % "chị Hằng", self.FINISH, send, self.FINISH]
        ep = self.run_one("train_0001", script, recover=True, recover_auto=True)
        self.assertEqual(ep.env.get_tool_calls()[:2], [{"name": "contacts.search", "arguments": {"query": "chị Hằng"}},
                                                       {"name": "contacts.search", "arguments": {"query": "Hằng"}}])
        self.assertEqual(ep.env.get_tool_calls()[2]["name"], "sms.send")
        self.assertEqual([(e["kind"], e.get("auto")) for e in ep.events],
                         [("call", None), ("call", "title"), ("call", None), ("finish", None)])
        self.assertTrue(any("lần tìm lại này có kết quả" in m["content"] for m in ep.messages))
        self.assertFalse(any(m["role"] == "assistant" and '"finish"' in m["content"] for m in ep.messages[:-1]))
        # off: the stop is only sent back with a note, as before
        ep = self.run_one("train_0001", script, recover=True)
        self.assertEqual([e["kind"] for e in ep.events][:2], ["call", "reject"])

    def test_agent_lists_everything(self):
        ep = self.run_one("train_0001", [self.SEARCH % "Lann", self.FINISH, self.SEND, self.FINISH],
                          recover=True, recover_auto=True)
        self.assertEqual([c["arguments"] for c in ep.env.get_tool_calls()[:2]], [{"query": "Lann"}, {}])
        self.assertEqual(ep.events[1].get("auto"), "all")
        self.assertEqual(ep.env.get_tool_calls()[2]["name"], "sms.send")
        self.assertTrue(any("đây là cả danh sách" in m["content"] for m in ep.messages))
        # a title, then the list: both are the agent's, the model only sees the results
        ep = self.run_one("train_0001", [self.SEARCH % "chị Hoa", self.FINISH, self.FINISH],
                          recover=True, recover_auto=True)
        self.assertEqual([c["arguments"] for c in ep.env.get_tool_calls()], [{"query": "chị Hoa"}, {"query": "Hoa"}, {}])
        self.assertEqual([e.get("auto") for e in ep.events], [None, "title", "all", None])
        self.assertEqual(ep.stop_reason, "finish")
        # an ask is held the same way; at most recover_calls such calls per sample
        ep = self.run_one("train_0001", [self.SEARCH % "chị Hoa", self.ASK % "contact_id"],
                          recover=True, recover_auto=True, recover_calls=1)
        self.assertEqual(len(ep.env.get_tool_calls()), 3)                   # search, the agent's one call, the ask
        self.assertEqual(ep.stop_reason, "ask")

    def test_agent_calls_keep_the_thought_format(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, prompt_cfg=PromptConfig(thoughts=True),
                        agent_cfg=AgentConfig(recover=True, recover_auto=True))
        from lifemate.agent import Agent
        ep = build_episode(rc, self.by_id["train_0001"])
        Agent(ScriptedBackend({"train_0001": [self.SEARCH % "Lann", self.FINISH, self.FINISH]}), rc.agent_cfg).run([ep])
        own = [m["content"] for m in ep.messages if m["role"] == "assistant"][1]
        self.assertTrue(own.startswith('{"thought": "Tìm theo điều kiện đó không ra'))
        self.assertEqual(parse_action(own)["arguments"], {})

    def test_model_rewrites_a_search(self):
        from lifemate.recovery import retry_prefix, retry_thought
        again = lambda d: '"date": "%s"}}' % d          # what the model writes after the forced start
        kw = dict(recover=True, recover_force=True, recover_force_k=3)
        script = [self.NEW, self.LIST % "2026-10-22", self.FINISH, again("2026-10-21"), again("2026-10-22"),
                  again("2026-10-21"), self.FINISH]
        ep = self.run_one("train_0003", script, **kw)
        self.assertEqual([c["arguments"].get("date") for c in ep.env.get_tool_calls()], [None, "2026-10-22", "2026-10-21"])
        self.assertEqual([e["kind"] for e in ep.events], ["call", "call", "hold", "call", "finish"])
        self.assertTrue(ep.events[3].get("forced") and not ep.events[3].get("empty"))
        self.assertEqual(ep.events[2]["stop"], "finish")
        # nothing new written (the same call, no JSON): the stop the model had sent stands
        ep = self.run_one("train_0003", [self.LIST % "2026-10-22", self.FINISH, again("2026-10-22"), "x", "y"], **kw)
        self.assertEqual(len(ep.env.get_tool_calls()), 1)
        self.assertEqual([e["kind"] for e in ep.events], ["call", "hold", "retry_none", "finish"])
        # once per failed search: the new call finds nothing either, the next stop is final
        ep = self.run_one("train_0003", [self.LIST % "2026-10-22", self.FINISH, again("2026-10-23"), "x", "y",
                                         self.FINISH, self.LIST % "2026-10-24"], **kw)
        self.assertEqual(len(ep.env.get_tool_calls()), 2)
        self.assertEqual([e["kind"] for e in ep.events], ["call", "hold", "call", "finish"])
        # the forced start is what action_text writes, and says what to change
        cal = self.cat.tools["calendar.list_events"]
        th = retry_thought(cal, {"date": "2026-10-22"}, {}, "")
        self.assertIn("bảng ngày", th)
        pre = retry_prefix(th, "read", "calendar.list_events")
        self.assertEqual(parse_action(pre + again("2026-10-21")),
                         {"thought": th, "type": "read", "name": "calendar.list_events", "arguments": {"date": "2026-10-21"}})
        self.assertEqual(action_text(parse_action(pre + again("2026-10-21"))), pre + again("2026-10-21"))
        self.assertEqual(retry_prefix(None, "read", "a.b"), '{"type": "read", "name": "a.b", "arguments": {')

    def test_retry_thought_names_the_place(self):
        from lifemate.recovery import retry_thought
        raw = {"apps": [{"id": "trips", "tools": [{"name": "trips.search", "kind": "read", "parameters": {
            "type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                             "date": {"type": "string"}},
            "required": ["origin", "destination", "date"]}}]}]}
        spec = Catalog(raw).tools["trips.search"]
        ctx = {"user_profile": {"city": "Đà Nẵng"}}
        args = {"origin": "HAN", "destination": "SGN", "date": "2026-12-08"}
        th = retry_thought(spec, args, ctx, "đặt cho tôi một vé đi hà nội ngày tám tháng mười hai")
        for part in ("nơi họ đang ở (Đà Nẵng)", "không đổi thành mã", "gọi lại trips.search"):
            self.assertIn(part, th)
        self.assertNotIn("bảng ngày", th)                 # the place is what is wrong: the day is left alone
        # the user named the place the model used: nothing is said about where they are, the day is worth a look
        th = retry_thought(spec, {**args, "origin": "Hà Nội"}, ctx, "đặt vé từ hà nội đi sài gòn")
        self.assertNotIn("đang ở", th)
        self.assertIn("bảng ngày", th)

    def test_safe_vote(self):
        a, b, c = self.SEARCH % "Lan", self.SEARCH % "Minh", self.SEARCH % "Hằng"
        end = [self.FINISH] * 5
        kw = dict(vote_k=5, vote_safe=True)
        # a stop does not replace a call, however many sampled candidates say so
        ep = self.run_one("train_0001", [a, self.FINISH, self.FINISH, self.FINISH, self.FINISH] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["arguments"], {"query": "Lan"})
        self.assertEqual(ep.events[0]["vote"], {"k": 5, "top": 1, "greedy": True, "over": "finish",
                                                "split": [["read", 1, 1], ["finish", 4, 1]]})
        ask = self.ASK % "content"
        ep = self.run_one("train_0001", [a, ask, ask, ask, ask] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["name"], "contacts.search")
        # another call needs more than half of the candidates
        ep = self.run_one("train_0001", [a, b, b, c, self.FINISH] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["arguments"], {"query": "Lan"})
        ep = self.run_one("train_0001", [a, b, b, b, c] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["arguments"], {"query": "Minh"})
        self.assertEqual(ep.events[0]["vote"], {"k": 5, "top": 3, "greedy": False,
                                                "split": [["read", 1, 1], ["read", 3, 1], ["read", 1, 1]]})
        # a stop gives way to a call the majority wants, and one kind of stop stays what it is
        ep = self.run_one("train_0001", [self.FINISH, b, b, b, c] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["arguments"], {"query": "Minh"})
        ep = self.run_one("train_0001", [ask, self.FINISH, self.FINISH, self.FINISH, self.FINISH], **kw)
        self.assertEqual(ep.stop_reason, "ask")
        # an action the guard would reject: any read that would run takes its place, a stop does not
        bad = self.SEND.replace("c_lan", "c_999")
        ep = self.run_one("train_0001", [bad, self.FINISH, a, self.FINISH, self.FINISH] + end, **kw)
        self.assertEqual(ep.env.get_tool_calls()[0]["name"], "contacts.search")
        self.assertFalse(any(e["kind"] == "reject" for e in ep.events))
        ep = self.run_one("train_0001", [bad] + [self.FINISH] * 4 + end, **kw)
        self.assertEqual(ep.events[0]["kind"], "reject")
        # the plain count (v33) for comparison
        ep = self.run_one("train_0001", [a, self.FINISH, self.FINISH, self.FINISH, self.FINISH], vote_k=5)
        self.assertEqual(ep.env.get_tool_calls(), [])

    def test_vote_only_on_the_first_steps(self):
        a, b = self.SEARCH % "Lan", self.SEARCH % "Minh"
        # step 0: three candidates (b wins); step 1 and later: one
        ep = self.run_one("train_0001", [a, b, b, self.SEND.replace("c_lan", "c_minh"), self.FINISH],
                          vote_k=3, vote_max_step=1)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "sms.send"])
        self.assertIn("vote", ep.events[0])
        self.assertNotIn("vote", ep.events[1])

    def test_ask_for_a_parameter_the_tool_does_not_have(self):
        from lifemate.guard import ask_param
        on = GuardConfig(fix_ask_params=True)
        # an ID under another name: the one ID the tool needs is meant
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", self.ASK % "chat_id"], guard=on)
        self.assertEqual(ep.env.get_tool_calls()[-1]["arguments"], {"tool_name": "sms.send", "params": ["contact_id"]})
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", self.ASK % "chat_id", self.FINISH])
        self.assertIn("không có tham số", ep.events[1]["error"])
        self.assertNotIn("tham số của sms.send", ep.events[1]["error"])
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", self.ASK % "mau_sac", self.FINISH], guard=on)
        self.assertIn("tham số của sms.send: ", ep.events[1]["error"])
        raw = {"apps": [{"id": "food", "tools": [{"name": "food.order", "kind": "write", "parameters": {
            "type": "object", "properties": {
                "shop_id": {"type": "string", "x-ref": "food.shops"},
                "items": {"type": "array", "items": {"type": "object", "properties": {
                    "item_id": {"type": "string"}, "quantity": {"type": "integer"}}}},
                "voucher_id": {"type": "string", "x-ref": "food.vouchers"}},
            "required": ["shop_id", "items"]}}]},
            {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "kind": "meta", "parameters": {
                "type": "object", "properties": {"tool_name": {"type": "string"},
                                                 "params": {"type": "array", "items": {"type": "string"}}},
                "required": ["tool_name", "params"]}}]}]}
        cat = Catalog(raw)
        order = cat.tools["food.order"]
        self.assertEqual(ask_param(order, "quantity"), ("items", "inside"))
        self.assertEqual(ask_param(order, "restaurant_id"), ("shop_id", "id"))   # voucher_id is not needed
        self.assertIsNone(ask_param(order, "note"))
        env = type("Env", (), {"validate": lambda self, n, a: {"ok": True}})()
        g = Guard(cat, list(cat.tools), {}, on)
        st = g.init_state(type("S", (), {"history": []})())
        act = {"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "food.order", "params": ["quantity"]}}
        d = g.check(act, env, st)
        self.assertEqual(d.kind, "reject")                       # once: where to look instead of asking
        self.assertIn("nằm trong items", d.message)
        d = g.check(act, env, st)
        self.assertEqual((d.kind, d.action["arguments"]["params"]), ("execute", ["items"]))

    def test_trace_stats_count_the_new_events(self):
        from lifemate.tracestats import format_stats, stats
        ep = self.run_one("train_0001", [self.SEARCH % "chị Hoa", self.FINISH, self.FINISH],
                          recover=True, recover_auto=True)
        s = stats([ep.result()], self.cat)
        self.assertEqual(s["counts"]["searches by the agent (title)"], 1)
        self.assertEqual(s["counts"]["searches by the agent (all)"], 1)
        self.assertEqual(s["counts"]["... that found something"], 1)
        self.assertEqual(s["failed_calls"]["next_event"], {"search by the agent": 2})
        a = self.SEARCH % "Lan"
        ep = self.run_one("train_0001", [a] + [self.FINISH] * 9, vote_k=5, vote_safe=True)
        s = stats([ep.result()], self.cat)
        self.assertEqual(s["votes"]["kept_against_the_count"], {"finish": 1})
        self.assertIn("kept although the plain count", format_stats(s))

    def test_effect_counts_only_where_an_option_acted(self):
        from lifemate.effect import effect, format_effect, touched
        ep = self.run_one("train_0001", [self.SEARCH % "Lann", self.FINISH, self.SEND, self.FINISH],
                          recover=True, recover_auto=True)
        self.assertEqual(touched(ep.result()), ["search by the agent"])
        ep2 = self.run_one("train_0001", [self.SEARCH % "Lan", self.ASK % "chat_id"],
                           guard=GuardConfig(fix_ask_params=True))
        self.assertEqual(touched(ep2.result()), ["ask parameter"])
        plain = self.run_one("train_0003", [self.NEW, self.FINISH])
        self.assertEqual(touched(plain.result()), [])
        per = lambda sc: {"score": 50.0, "per_sample": [
            {"id": "train_0001", "score": sc[0], "hash_ok": sc[0] > 0}, {"id": "train_0003", "score": sc[1], "hash_ok": True}]}
        r = effect(per((0.0, 1.0)), per((1.0, 0.5)), [ep.result(), plain.result()])
        self.assertEqual(r["acted"], {"samples": 1, "points": 50.0, "up": 1, "down": 0, "hash": 1})
        self.assertEqual(r["not_touched"], {"samples": 1, "points": -25.0, "up": 0, "down": 1, "hash": 0})
        self.assertIn("search by the agent", format_effect(r, "x"))

    def test_gold_stays_at_100_with_everything_on(self):
        from lifemate.cli import _load_cfg
        g = GuardConfig.from_json({**_load_cfg(None)["guard"], "fix_ask_params": True, "block_failed_repeats": True})
        a = AgentConfig(recover=True, notice_repeat=True, recover_auto=True, recover_force=True, vote_k=5,
                        vote_safe=True, vote_max_step=3)
        for thoughts in (False, True):
            rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, guard_cfg=g, agent_cfg=a,
                            prompt_cfg=PromptConfig(thoughts=thoughts, date_hints=True))
            res = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, batch_size=4, log_every=0)
            self.assertEqual(Scorer(self.sp, self.cat).score(self.rows(res), self.samples)["score"], 100.0)

    def test_vllm_computes_a_shared_prompt_once(self):
        import types
        from unittest import mock
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **kw: kw
        from lifemate.llm import GenRequest, VLLMBackend
        b = object.__new__(VLLMBackend)
        b.max_model_len, b.lora_req = 100, None
        b.tok = types.SimpleNamespace(encode=lambda t, add_special_tokens=False: t.split())
        calls = []

        def gen(prompts, params, **kw):
            calls.append((list(prompts), params))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"o{i}")]) for i in range(len(prompts))]

        b.llm = types.SimpleNamespace(generate=gen)
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            got = b.generate([GenRequest(None, "p q", "", 0.0, 64), GenRequest(None, "p q", "", 0.7, 64, 1),
                              GenRequest(None, "p q", "", 0.7, 64, 2), GenRequest(None, "r", "", 0.0, 64)])
        self.assertEqual(got, ["o0", "o1", "o2", "o3"])
        self.assertEqual(calls[0], (["p q"], {"temperature": 0.0, "max_tokens": 1}))
        self.assertEqual(len(calls[1][0]), 4)
        with mock.patch.dict(sys.modules, {"vllm": fake}):          # no prompt twice: one call, as before
            calls.clear()
            b.generate([GenRequest(None, "p", "", 0.0, 64), GenRequest(None, "r", "", 0.0, 64)])
        self.assertEqual(len(calls), 1)


class RecordingBackend(ScriptedBackend):
    """Scripted answers, and the prompts each adapter was given."""
    needs_prompt = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prompts = []

    def render(self, messages):
        return "\n".join(m["content"] for m in messages)

    def generate(self, reqs):
        self.prompts += [(r.adapter, r.prompt) for r in reqs]
        return super().generate(reqs)


class TestConsult(Base):
    """v36: a second adapter is asked at the first stop without a write (AgentConfig.consult)."""
    SEARCH, SEND, FINISH = TestRecovery.SEARCH, TestRecovery.SEND, TestRecovery.FINISH
    ASK = TestActiveRecovery.ASK
    THINK_SEND = ('{"thought": "đã có c_lan", "type": "write", "name": "sms.send", '
                  '"arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn"}}')

    def run_one(self, main, answers, sid="train_0001", **agent):
        from lifemate.agent import Agent
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, agent_cfg=AgentConfig(**agent))
        ep = build_episode(rc, self.by_id[sid])
        be = ScriptedBackend({sid: list(main)}, {sid: list(answers)})
        Agent(be, rc.agent_cfg).run([ep])
        return ep, be

    def names(self, ep):
        return [c["name"] for c in ep.env.get_tool_calls()]

    def test_off_by_default(self):
        ep, be = self.run_one([self.SEARCH % "lan", self.FINISH], [self.THINK_SEND])
        self.assertEqual(self.names(ep), ["contacts.search"])
        self.assertEqual(be.consult_calls, {})
        self.assertFalse(any(e["kind"] == "consult" or e.get("consult") for e in ep.events))

    def test_write_taken_at_a_finish(self):
        ep, be = self.run_one([self.SEARCH % "lan", self.FINISH, self.FINISH], [self.THINK_SEND], consult="write")
        self.assertEqual(self.names(ep), ["contacts.search", "sms.send"])
        self.assertEqual(ep.events[1]["consult"], {"on": "finish", "mode": "write"})
        self.assertEqual(ep.events[1]["thought"], "đã có c_lan")
        # the main model goes on after it and its history keeps its own format (no thought)
        self.assertEqual((be.calls["train_0001"], be.consult_calls["train_0001"]), (3, 1))
        self.assertEqual((ep.events[-1]["kind"], ep.stop_reason), ("finish", "finish"))
        last = [m["content"] for m in ep.messages if m["role"] == "assistant"][-1]
        self.assertTrue(last.startswith('{"type": "write", "name": "sms.send"'))
        self.assertEqual(ep.driver, "main")

    def test_write_taken_instead_of_an_ask(self):
        ep, _ = self.run_one([self.SEARCH % "lan", self.ASK % "contact_id", self.FINISH], [self.THINK_SEND],
                             consult="write")
        self.assertEqual(self.names(ep), ["contacts.search", "sms.send"])      # the ask never ran
        self.assertEqual(ep.events[1]["consult"]["on"], "ask")
        self.assertEqual(ep.stop_reason, "finish")

    def test_the_stop_stands_when_nothing_is_taken(self):
        # write mode takes no read
        ep, _ = self.run_one([self.FINISH], [self.SEARCH % "lan"], consult="write")
        self.assertEqual(ep.env.get_tool_calls(), [])
        self.assertEqual([(e["kind"], e.get("took"), e.get("type")) for e in ep.events],
                         [("consult", False, "read"), ("finish", None, None)])
        # an ask that waited runs as it would have
        ep, _ = self.run_one([self.SEARCH % "lan", self.ASK % "contact_id"], [self.FINISH], consult="write")
        self.assertEqual(self.names(ep), ["contacts.search", "sys.ask_missing_info"])
        self.assertEqual(ep.stop_reason, "ask")
        # nothing the guard would reject, and one look only
        ep, be = self.run_one([self.FINISH], ['{"type": "write", "name": "foo.bar", "arguments": {}}', self.THINK_SEND],
                              consult="handover")
        self.assertEqual(ep.env.get_tool_calls(), [])
        self.assertEqual(be.consult_calls["train_0001"], 1)
        self.assertEqual(ep.stop_reason, "finish")

    def test_handover(self):
        ep, be = self.run_one([self.FINISH], [self.SEARCH % "lan", self.THINK_SEND, self.FINISH], consult="handover")
        self.assertEqual(self.names(ep), ["contacts.search", "sms.send"])
        self.assertEqual((ep.driver, be.calls["train_0001"]), ("consult", 1))
        self.assertEqual(ep.events[0]["consult"]["on"], "finish")
        self.assertEqual([e.get("by") for e in ep.events[1:]], ["consult", "consult"])
        # that model goes on, so its thoughts stay in the history
        self.assertIn('"thought": "đã có c_lan"', [m["content"] for m in ep.messages if m["role"] == "assistant"][-1])

    def test_only_before_the_first_write_and_in_time(self):
        ep, be = self.run_one([self.SEARCH % "lan", self.SEND, self.FINISH], [self.THINK_SEND], consult="write")
        self.assertEqual((be.consult_calls, len(ep.env.get_tool_calls())), ({}, 2))
        ep, be = self.run_one([self.FINISH], [self.THINK_SEND], consult="write", consult_until_s=1e-9)
        self.assertEqual((be.consult_calls, ep.env.get_tool_calls()), ({}, []))

    def test_consult_prompt_has_its_own_format(self):
        from lifemate.agent import Agent
        from lifemate.prompt import THOUGHT_MARK
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, agent_cfg=AgentConfig(consult="write"),
                        consult_prompt_cfg=PromptConfig(thoughts=True))
        ep = build_episode(rc, self.by_id["train_0001"])
        be = RecordingBackend({"train_0001": [self.SEARCH % "lan", self.FINISH, self.FINISH]},
                              {"train_0001": [self.THINK_SEND]})
        Agent(be, rc.agent_cfg).run([ep])
        main = [p for ad, p in be.prompts if ad is None]
        cons = [p for ad, p in be.prompts if ad == "consult"]
        self.assertEqual((len(main), len(cons)), (3, 1))
        self.assertIn(THOUGHT_MARK, cons[0])
        self.assertFalse(any(THOUGHT_MARK in p for p in main))
        self.assertIn('"name": "contacts.search"', cons[0])        # the same steps after its own first messages

    def test_batched_run(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, agent_cfg=AgentConfig(consult="write"))
        s1, s2 = self.by_id["train_0001"], self.by_id["train_0002"]
        be = ScriptedBackend({s1.id: [self.SEARCH % "Lan", self.FINISH, self.FINISH], s2.id: [self.FINISH]},
                             {s1.id: [self.THINK_SEND], s2.id: [self.FINISH]})
        res = {r["id"]: r for r in run_split(rc, be, [s1, s2], batch_size=2, log_every=0)}
        self.assertEqual([c["name"] for c in res[s1.id]["tool_calls"]], ["contacts.search", "sms.send"])
        self.assertEqual(res[s2.id]["tool_calls"], [])
        self.assertEqual(Scorer(self.sp, self.cat).score(self.rows([res[s1.id]]), [s1], True)["score"], 100.0)

    def test_effect_and_stats(self):
        from lifemate.effect import touched
        from lifemate.tracestats import stats
        ep, _ = self.run_one([self.SEARCH % "lan", self.FINISH, self.FINISH], [self.THINK_SEND], consult="write")
        self.assertEqual(touched(ep.result()), ["consult taken at finish"])
        self.assertEqual(stats([ep.result()], self.cat)["counts"].get("consult at finish: write taken"), 1)
        ep, _ = self.run_one([self.FINISH], [self.SEARCH % "lan"], consult="write")
        self.assertEqual(touched(ep.result()), [])                        # the stop stood: nothing changed
        self.assertEqual(stats([ep.result()], self.cat)["counts"].get("consult at finish: not taken (read)"), 1)

    def test_vllm_two_adapters(self):
        import types
        from unittest import mock
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **kw: kw
        from lifemate.llm import GenRequest, VLLMBackend
        b = object.__new__(VLLMBackend)
        b.max_model_len, b.lora_req, b.consult_req = 100, "MAIN", "CONSULT"
        b.tok = types.SimpleNamespace(encode=lambda t, add_special_tokens=False: t.split())
        calls = []

        def gen(prompts, params, lora_request=None, **kw):
            calls.append((list(prompts), lora_request))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"o{i}")]) for i in range(len(prompts))]

        b.llm = types.SimpleNamespace(generate=gen)
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            got = b.generate([GenRequest(None, "p", "", 0.0, 64), GenRequest(None, "p", "", 0.7, 64, 1),
                              GenRequest(None, "p", "", 0.0, 64, adapter="consult")])
        self.assertEqual(got, ["o0", "o1", "o2"])
        self.assertEqual(calls[0], (["p"], ["MAIN"]))                   # only the main prompt is there twice
        self.assertEqual(calls[1], (["p", "p", "p"], ["MAIN", "MAIN", "CONSULT"]))
        self.assertTrue(b.has_adapter("consult"))
        with mock.patch.dict(sys.modules, {"vllm": fake}):              # one adapter for all: passed as before
            calls.clear()
            b.generate([GenRequest(None, "q", "", 0.0, 64)])
        self.assertEqual(calls, [(["q"], "MAIN")])

    def test_predict_finds_two_adapters(self):
        import tempfile
        from lifemate.predict import PredictConfig, resolve_adapters
        d = Path(tempfile.mkdtemp())
        for n in ("main", "consult"):
            (d / "adapters" / n).mkdir(parents=True)
            (d / "adapters" / n / "adapter_config.json").write_text("{}")
        c = resolve_adapters(PredictConfig(llm=str(d)))
        self.assertEqual((c.lora, c.consult_lora), (str(d / "adapters" / "main"), str(d / "adapters" / "consult")))
        self.assertEqual(resolve_adapters(PredictConfig(llm=str(d), lora="x")).lora, "x")
        plain = resolve_adapters(PredictConfig(llm=tempfile.mkdtemp()))       # a merged model: nothing else
        self.assertEqual((plain.lora, plain.consult_lora), (None, None))
        self.assertEqual(PredictConfig().override(["--consult-lora", "/a"]).consult_lora, "/a")

    def test_cli_consult_flags(self):
        import contextlib
        import io
        import tempfile
        from lifemate import cli
        d = Path(tempfile.mkdtemp())
        (d / "lifemate_prompt.json").write_text('{"thoughts": true}')
        out = d / "sub.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["infer", "--data", str(DATA), "--split", "train", "--backend", "empty", "--text", "oracle",
                      "--consult", "handover", "--consult-lora", str(d), "--limit", "2", "--out", str(out)])
        self.assertIn("consult=handover thoughts=True", buf.getvalue())
        self.assertEqual(len(read_jsonl(out)), 2)


class TestSelfConsult(unittest.TestCase):
    """v38: no second adapter, the main model is asked again in the other prompt format."""

    def test_backend_asks_the_same_model(self):
        import types
        from unittest import mock
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **kw: kw
        from lifemate.llm import GenRequest, VLLMBackend
        for main in ("MAIN", None):                    # an adapter, or a merged model (no LoRA at all)
            b = object.__new__(VLLMBackend)
            b.max_model_len, b.lora_req, b.consult_req, b.consult_self = 100, main, None, True
            b.tok = types.SimpleNamespace(encode=lambda t, add_special_tokens=False: t.split())
            calls = []

            def gen(prompts, params, lora_request=None, **kw):
                calls.append((list(prompts), lora_request))
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="o")]) for _ in prompts]

            b.llm = types.SimpleNamespace(generate=gen)
            with mock.patch.dict(sys.modules, {"vllm": fake}):
                b.generate([GenRequest(None, "p", "", 0.0, 64), GenRequest(None, "q", "", 0.0, 64, adapter="consult")])
            self.assertTrue(b.has_adapter("consult"))
            self.assertEqual(calls, [(["p", "q"], main)])     # one model for both, passed as a single request
        b.consult_self = False
        self.assertFalse(b.has_adapter("consult"))

    def test_consult_prompt(self):
        import tempfile
        from lifemate.prompt import consult_prompt
        c, src = consult_prompt(PromptConfig(thoughts=True))
        self.assertEqual((c.thoughts, src), (False, None))                      # the main model again: other format
        self.assertTrue(consult_prompt(PromptConfig(thoughts=False))[0].thoughts)
        self.assertTrue(consult_prompt(PromptConfig(thoughts=True), thoughts="on")[0].thoughts)
        d = Path(tempfile.mkdtemp())
        (d / "lifemate_prompt.json").write_text('{"thoughts": true}')
        c, src = consult_prompt(PromptConfig(thoughts=False, date_hints=True), str(d))
        self.assertEqual((c.thoughts, c.date_hints, src.name), (True, True, "lifemate_prompt.json"))   # the adapter's
        self.assertFalse(consult_prompt(PromptConfig(), str(d), "off")[0].thoughts)

    def test_cli_flags(self):
        import contextlib
        import io
        import tempfile
        from lifemate import cli
        out = Path(tempfile.mkdtemp()) / "sub.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["infer", "--data", str(DATA), "--split", "train", "--backend", "empty", "--text", "oracle",
                      "--consult", "write", "--consult-self", "--limit", "2", "--out", str(out)])
            cli.main(["infer", "--data", str(DATA), "--split", "train", "--backend", "empty", "--text", "oracle",
                      "--consult", "write", "--consult-self", "--consult-thoughts", "off", "--limit", "1",
                      "--out", str(out)])
        self.assertIn("consult=write thoughts=True (the main model again)", buf.getvalue())
        self.assertIn("consult=write thoughts=False (the main model again)", buf.getvalue())
        self.assertEqual(len(read_jsonl(out)), 1)


class TestSoup(unittest.TestCase):
    """v37: two adapters of the same base model summed with weights, as one adapter."""
    KEY = "base_model.model.model.layers.0.self_attn.{}"

    def make(self, d, seed, r=2, alpha=4, thoughts=None, modules=("q_proj", "down_proj")):
        import torch
        from safetensors.torch import save_file
        g = torch.Generator().manual_seed(seed)
        w = {}
        for m in modules:
            w[self.KEY.format(m) + ".lora_A.weight"] = torch.randn(r, 6, generator=g)
            w[self.KEY.format(m) + ".lora_B.weight"] = torch.randn(5, r, generator=g)
        d.mkdir(parents=True)
        save_file(w, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "target_modules": list(modules),
                                                           "base_model_name_or_path": "/x/base-llm"}))
        if thoughts is not None:
            (d / "lifemate_prompt.json").write_text(json.dumps({"thoughts": thoughts}))
        return w

    @staticmethod
    def delta(w, key, scale):
        return scale * w[key + ".lora_B.weight"] @ w[key + ".lora_A.weight"]

    def test_soup_is_the_weighted_sum(self):
        import tempfile
        import torch
        from safetensors.torch import load_file
        from lifemate.soup_lora import soup
        d = Path(tempfile.mkdtemp())
        wa = self.make(d / "a", 1, thoughts=False)
        wb = self.make(d / "b", 2, alpha=8, thoughts=True)          # another scale: 8 / 2 = 4, a has 4 / 2 = 2
        info = soup(str(d / "a"), str(d / "b"), 0.7, str(d / "s"), thoughts="b")
        ws = load_file(str(d / "s" / "adapter_model.safetensors"))
        cfg = json.loads((d / "s" / "adapter_config.json").read_text())
        self.assertEqual((cfg["r"], cfg["lora_alpha"], info["rank"]), (4, 8.0, 4))   # a's scale kept: 8 / 4 = 2
        for m in ("q_proj", "down_proj"):
            k = self.KEY.format(m)
            want = 0.7 * self.delta(wa, k, 2.0) + 0.3 * self.delta(wb, k, 4.0)
            self.assertTrue(torch.allclose(self.delta(ws, k, 2.0), want, atol=1e-5))
        self.assertTrue(json.loads((d / "s" / "lifemate_prompt.json").read_text())["thoughts"])
        # lam 1 is adapter a alone; the prompt format can be set
        soup(str(d / "a"), str(d / "b"), 1.0, str(d / "t"), thoughts="off")
        wt = load_file(str(d / "t" / "adapter_model.safetensors"))
        k = self.KEY.format("q_proj")
        self.assertTrue(torch.allclose(self.delta(wt, k, 2.0), self.delta(wa, k, 2.0), atol=1e-6))
        self.assertFalse(json.loads((d / "t" / "lifemate_prompt.json").read_text())["thoughts"])

    def test_soup_command_and_mismatch(self):
        import contextlib
        import io
        import tempfile
        from lifemate import cli
        d = Path(tempfile.mkdtemp())
        self.make(d / "a", 1, thoughts=False)
        self.make(d / "b", 2, thoughts=True)
        self.make(d / "c", 3, modules=("q_proj", "v_proj"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["soup-lora", "--a", str(d / "a"), "--b", str(d / "b"), "--lam", "0.5", "--out", str(d / "s")])
        self.assertIn("'rank': 4", buf.getvalue())
        self.assertFalse(json.loads((d / "s" / "lifemate_prompt.json").read_text())["thoughts"])   # like a
        with self.assertRaises(SystemExit):
            cli.main(["soup-lora", "--a", str(d / "a"), "--b", str(d / "c"), "--out", str(d / "x")])
        with self.assertRaises(SystemExit):
            cli.main(["soup-lora", "--a", str(d / "a"), "--b", str(d / "b"), "--lam", "1.5", "--out", str(d / "y")])


class TestRefIdAndForcedLookup(Base):
    """v39: guard.prefer_ref_id (a raw value a record holds -> that record's ID) and agent.force_lookup (the call the
    guard named is started for the model when it asks again until the limit or stops instead)."""
    SEARCH, SEND, FINISH = TestRecovery.SEARCH, TestRecovery.SEND, TestRecovery.FINISH
    ASK = '{"type": "ask", "name": "sys.ask_missing_info", "arguments": {"tool_name": "%s", "params": ["%s"]}}'
    LAN = {"ok": True, "results": [{"id": "c_lan", "name": "Chị Lan", "phone": "0901111222"}]}
    BY_PHONE = {"type": "write", "name": "sms.send", "arguments": {"phone_number": "0901111222", "content": "xin chào"}}

    def _guard(self, sid, **cfg):
        s = self.by_id[sid]
        g = Guard(self.cat, self.cat.allowed_tools(s.installed_tool_names()), s.context, GuardConfig(**cfg), self.fb)
        return g, g.init_state(s), make_env(self.sp, s.id)

    def run_one(self, sid, script, guard=None, thoughts=False, **agent):
        from lifemate.agent import Agent
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb, guard_cfg=guard or GuardConfig(),
                        prompt_cfg=PromptConfig(thoughts=thoughts), agent_cfg=AgentConfig(**agent))
        ep = build_episode(rc, self.by_id[sid])
        Agent(ScriptedBackend({sid: list(script)}), rc.agent_cfg).run([ep])
        return ep

    def test_note_records(self):
        from lifemate.guard import note_records
        out = {}
        note_records("contacts", {"ok": True, "results": [{"id": "c_1", "name": "Lan", "phone": "0901", "rel": "bạn"},
                                                          {"id": "c_2", "name": "Minh", "phone": "0902", "rel": "bạn"}]},
                     out)
        self.assertEqual(out["contacts"]["0901"], "c_1")
        self.assertEqual(out["contacts"]["Minh"], "c_2")
        self.assertIsNone(out["contacts"]["bạn"])                     # two records share it: no single ID
        self.assertNotIn("c_1", out["contacts"])                      # the ID itself is not a value of the record
        note_records("contacts", {"item": {"id": "c_3", "phone": "0901"}}, out)
        self.assertIsNone(out["contacts"]["0901"])

    def test_raw_value_becomes_the_record_id(self):
        g, st, env = self._guard("train_0001", prefer_ref_id=True)
        g.record({"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}, self.LAN, st)
        d = g.check(self.BY_PHONE, env, st)
        self.assertEqual(d.kind, "execute", d.message)
        self.assertEqual(list(d.action["arguments"].items()), [("contact_id", "c_lan"), ("content", "xin chào")])
        self.assertEqual(d.fixes, ["phone_number->contact_id"])
        # a number no record holds (the user read it out), an ID already there, the flag off: left alone
        g, st, env = self._guard("train_0001", prefer_ref_id=True)
        g.record({"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}, self.LAN, st)
        other = {**self.BY_PHONE, "arguments": {"phone_number": "0909999999", "content": "xin chào"}}
        self.assertEqual(g.check(other, env, st).action["arguments"]["phone_number"], "0909999999")
        both = {**self.BY_PHONE, "arguments": {"contact_id": "c_lan", "phone_number": "0901111222", "content": "a"}}
        self.assertEqual(g.check(both, env, st).fixes, [])
        g, st, env = self._guard("train_0001")
        g.record({"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}, self.LAN, st)
        self.assertEqual(g.check(self.BY_PHONE, env, st).action["arguments"]["phone_number"], "0901111222")
        # a record of another app holding the number is not a contact
        g, st, env = self._guard("train_0001", prefer_ref_id=True)
        g.record({"type": "read", "name": "smart_home.get_status", "arguments": {}},
                 {"ok": True, "devices": [{"id": "light_living", "phone": "0901111222"}]}, st)
        self.assertEqual(g.check(self.BY_PHONE, env, st).fixes, [])

    def test_swap_is_marked_in_the_trace(self):
        from lifemate.effect import touched
        by_phone = json.dumps(self.BY_PHONE, ensure_ascii=False)
        ep = self.run_one("train_0001", [self.SEARCH % "Lan", by_phone, self.FINISH],
                          guard=GuardConfig(prefer_ref_id=True))
        self.assertEqual(ep.env.get_tool_calls()[1]["arguments"], {"contact_id": "c_lan", "content": "xin chào"})
        self.assertEqual(ep.events[1]["swap"], "phone_number->contact_id")
        self.assertIn("ID instead of a raw value", touched({"events": ep.events}))

    def test_named_lookup_forced_at_the_reject_limit(self):
        ask = self.ASK % ("sms.send", "contact_id")
        cont = ['"query": "Lan"}}'] * 5                     # what the model writes after the forced start
        script = [ask, ask, ask] + cont + [self.SEND, self.FINISH]
        guard = GuardConfig(ask_ids_after_read=True)
        off = self.run_one("train_0001", script, guard=guard)
        self.assertEqual(off.stop_reason, "rejected")
        self.assertEqual(off.env.get_tool_calls(), [])
        on = self.run_one("train_0001", script, guard=guard, force_lookup=True)
        self.assertEqual([c["name"] for c in on.env.get_tool_calls()], ["contacts.search", "sms.send"])
        self.assertEqual(on.env.get_tool_calls()[0]["arguments"], {"query": "Lan"})
        self.assertEqual([e["kind"] for e in on.events], ["reject"] * 3 + ["lookup_hold", "call", "call", "finish"])
        self.assertEqual(on.events[3], {"step": 0, "kind": "lookup_hold", "name": "contacts.search", "stop": "rejected"})
        self.assertEqual(on.events[4]["lookup"], "contacts.search")
        self.assertEqual(on.stop_reason, "finish")
        # nothing valid written after the start: the sample stops as it would have
        bad = self.run_one("train_0001", [ask, ask, ask] + ["x"] * 5, guard=guard, force_lookup=True)
        self.assertEqual(bad.stop_reason, "rejected")
        self.assertEqual([e["kind"] for e in bad.events][-2:], ["lookup_hold", "lookup_none"])
        self.assertEqual(bad.env.get_tool_calls(), [])

    def test_named_lookup_forced_at_a_finish(self):
        from lifemate.effect import touched
        ask = self.ASK % ("sms.send", "contact_id")
        cont = ['"query": "Lan"}}'] * 5
        guard = GuardConfig(ask_ids_after_read=True)
        ep = self.run_one("train_0001", [ask, self.FINISH] + cont + [self.SEND, self.FINISH], guard=guard,
                          force_lookup=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "sms.send"])
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "lookup_hold", "call", "call", "finish"])
        self.assertEqual(ep.events[1]["stop"], "finish")
        self.assertIn("call the guard named, forced", touched({"events": ep.events}))
        # nothing valid: the finish the model had sent stands
        ep = self.run_one("train_0001", [ask, self.FINISH] + ["x"] * 5, guard=guard, force_lookup=True)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "lookup_hold", "lookup_none", "finish"])
        self.assertEqual(ep.stop_reason, "finish")
        # once per sample: a second rejection that names a lookup, then a stop, are left alone
        bad_id = self.SEND.replace("c_lan", "c_xx")
        ep = self.run_one("train_0001", [ask, self.FINISH] + cont + [bad_id, self.FINISH], guard=guard,
                          force_lookup=True)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "lookup_hold", "call", "reject", "finish"])
        self.assertEqual(ep.stop_reason, "finish")

    def test_no_force_once_the_model_looked_it_up(self):
        ask = self.ASK % ("sms.send", "contact_id")
        guard = GuardConfig(ask_ids_after_read=True)
        ep = self.run_one("train_0001", [ask, self.SEARCH % "Lan", self.FINISH], guard=guard, force_lookup=True)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "call", "finish"])
        # a write after the rejection: the model went on with the request, its stop stands; a read of another app
        # does not count as the lookup
        status = '{"type": "read", "name": "smart_home.get_status", "arguments": {}}'
        light = '{"type": "write", "name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off"}}'
        ep = self.run_one("train_0001", [ask, status, light, self.FINISH], guard=guard, force_lookup=True)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "call", "call", "finish"])
        ep = self.run_one("train_0001", [ask, status, self.FINISH] + ["x"] * 5, guard=guard, force_lookup=True)
        self.assertEqual([e["kind"] for e in ep.events], ["reject", "call", "lookup_hold", "lookup_none", "finish"])

    def test_named_write_forced_when_only_optional_params_are_asked(self):
        ask = self.ASK % ("banking.transfer", "note")
        cont = ['"contact_id": "c_hang", "amount": 1200000}}'] * 5
        guard = GuardConfig(ask_needed_only=True)
        ep = self.run_one("train_0009", [self.SEARCH % "Hằng", ask, ask, ask] + cont + [self.FINISH], guard=guard,
                          force_lookup=True)
        self.assertEqual([c["name"] for c in ep.env.get_tool_calls()], ["contacts.search", "banking.transfer"])
        self.assertEqual(ep.events[4]["name"], "banking.transfer")
        self.assertEqual(ep.events[4]["stop"], "rejected")

    def test_acted_on_counts_each_sample_once(self):
        from lifemate.effect import acted_on
        rep = lambda xs: {"per_sample": [{"id": i, "score": s, "hash_ok": s > 0} for i, s in xs]}
        base = rep([("a", 0.0), ("b", 1.0), ("c", 0.5), ("d", 0.0)])
        run = rep([("a", 1.0), ("b", 0.5), ("c", 0.5), ("d", 1.0)])
        trace = [{"id": "a", "events": [{"kind": "lookup_hold"}, {"kind": "call", "lookup": "x", "swap": "p->id"}]},
                 {"id": "b", "events": [{"kind": "call", "swap": "p->id"}]},
                 {"id": "c", "events": [{"kind": "call"}]},
                 {"id": "d", "events": [{"kind": "call", "consult": {"on": "ask"}}]}]
        r = acted_on(base, run, trace, ("ID instead of a raw value", "call the guard named, forced"))
        self.assertEqual(r, {"samples": 2, "points": 12.5, "up": 1, "down": 1})     # (+1 - 0.5) / 4 samples

    def test_forced_start_keeps_the_thought_format(self):
        from lifemate.recovery import named_thought
        ask = self.ASK % ("sms.send", "contact_id")
        ep = self.run_one("train_0001", [ask, self.FINISH] + ['"query": "Lan"}}'] * 5, thoughts=True,
                          guard=GuardConfig(ask_ids_after_read=True), force_lookup=True)
        own = [m["content"] for m in ep.messages if m["role"] == "assistant"]
        th = named_thought("contacts.search", "read")
        self.assertTrue(any(c.startswith('{"thought": ' + json.dumps(th, ensure_ascii=False)) for c in own), own)
        self.assertIn("phải tra bằng contacts.search", th)


class TestActionsScorer(unittest.TestCase):
    def test_parse(self):
        a = parse_action('xx {"type": "write", "name": "sms.send", "arguments": {"content": "a"}} yy')
        self.assertEqual(a["name"], "sms.send")
        self.assertEqual(parse_action('{"type": "finish"}')["type"], "finish")
        self.assertIsNotNone(parse_action('{"type": "read", "name": "x.y", "arguments": {}'))
        self.assertIsNone(parse_action("không có json"))
        t = action_text({"type": "read", "name": "a.b", "arguments": {"q": "Lan"}})
        self.assertTrue(t.startswith('{"type": "read", "name": "a.b"'))

    def test_token_f1_and_matching(self):
        self.assertGreater(token_f1("tối nay em về muộn", "tối nay em về muộn nhé"), 0.6)
        self.assertLess(token_f1("đi chợ", "tối nay em về muộn"), 0.6)
        fz = {"sms.send": {"content"}}
        gold = [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "về muộn"}}] * 2
        pred = [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "về muộn nha"}}]
        self.assertEqual(max_matching(pred, gold, fz), 1)


class TestPipeline(Base):
    def test_oracle_100_and_empty_equals_oos(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb)
        res = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, batch_size=5, log_every=0)
        rep = Scorer(self.sp, self.cat).score(self.rows(res), self.samples)
        self.assertEqual(rep["score"], 100.0)
        empty = [{"id": s.id, "tool_calls": [], "state_hash": make_env(self.sp, s.id).get_state_hash()}
                 for s in self.samples]
        rep_e = Scorer(self.sp, self.cat).score(empty, self.samples)
        n_oos = rep["by_category"]["oos"]["n"]
        self.assertAlmostEqual(rep_e["score"], round(100 * n_oos / len(self.samples), 2))

    def test_deadline_only_for_single_runs(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb,
                        agent_cfg=AgentConfig(deadline_s=1e-9))
        res = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, batch_size=len(self.samples),
                        log_every=0)
        self.assertEqual(Scorer(self.sp, self.cat).score(self.rows(res), self.samples)["score"], 100.0)
        one = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples[:1], batch_size=1, log_every=0)[0]
        self.assertEqual(one["stop_reason"], "deadline")

    def test_type_scoring_with_oracle(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb,
                        agent_cfg=AgentConfig(use_type_scoring=True))
        res = run_split(rc, OracleBackend(self.by_id, self.cat), self.samples, batch_size=3, log_every=0)
        self.assertEqual(Scorer(self.sp, self.cat).score(self.rows(res), self.samples)["score"], 100.0)

    def test_guard_rejections_never_reach_env_log(self):
        s = self.by_id["train_0001"]
        script = {s.id: [
            '{"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn"}}',
            '{"type": "read", "name": "search", "arguments": {"query": "Lan"}}',
            '{"type": "read", "name": "contacts.search", "arguments": {"query": "Lan"}}',
            'lung tung',
            '{"type": "write", "name": "sms.send", "arguments": {"contact_id": "c_lan", "content": "tối nay em về muộn", "x": 1}}',
            '{"type": "finish"}']}
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb)
        res = run_split(rc, ScriptedBackend(script), [s], log_every=0)[0]
        names = [c["name"] for c in res["tool_calls"]]
        self.assertEqual(names, ["contacts.search", "sms.send"])
        self.assertEqual(Scorer(self.sp, self.cat).score(self.rows([res]), [s], True)["score"], 100.0)
        kinds = [e["kind"] for e in res["events"]]
        self.assertIn("reject", kinds)
        self.assertIn("cached", kinds)

    def test_guard_coercion_and_time_format(self):
        s = self.by_id["train_0007"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        g = Guard(self.cat, tools, s.context, GuardConfig(), self.fb)
        st = g.init_state(s)
        env = make_env(self.sp, s.id)
        st.seen.add("light_bedroom")
        d = g.check({"type": "write", "name": "smart_home.light_set",
                     "arguments": {"device_id": "light_bedroom", "power": "ON", "brightness": "50"}}, env, st)
        self.assertEqual(d.kind, "execute")
        self.assertEqual(d.action["arguments"], {"device_id": "light_bedroom", "power": "on", "brightness": 50})
        self.assertEqual(coerce_shape("2026-10-21 8:00", "datetime_tz", s.context), "2026-10-21T08:00:00+07:00")
        self.assertEqual(env.get_tool_calls(), [])

    def test_ask_guard(self):
        s = self.by_id["train_0004"]
        tools = self.cat.allowed_tools(s.installed_tool_names())
        g = Guard(self.cat, tools, s.context, GuardConfig(), self.fb)
        env = make_env(self.sp, s.id)
        st = g.init_state(s)
        bad = g.check({"type": "ask", "name": "sys.ask_missing_info",
                       "arguments": {"tool_name": "sms.send", "params": ["noi_dung"]}}, env, st)
        self.assertEqual(bad.kind, "reject")
        ok = g.check({"type": "ask", "name": "sys.ask_missing_info",
                      "arguments": {"tool_name": "send", "params": "content"}}, env, st)
        self.assertEqual(ok.kind, "execute")
        self.assertEqual(ok.action["arguments"], {"tool_name": "sms.send", "params": ["content"]})
        g2 = Guard(self.cat, tools, s.context, GuardConfig(ask_param_order="schema"), self.fb)
        two = g2.check({"type": "ask", "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "sms.send", "params": ["content", "contact_id"]}}, env, st)
        self.assertEqual(two.action["arguments"]["params"], ["contact_id", "content"])

    def test_sft_prompt_identical_to_inference(self):
        pool = FewShotPool.from_samples(self.samples, self.cat)
        pcfg = PromptConfig()
        rows, _ = build_rows(self.samples, self.cat, None, pool, pcfg, SFTConfig(n_aug=0))
        rc = RunContext(self.sp, self.cat, text_source="oracle", pool=pool, prompt_cfg=pcfg)
        for r in rows:
            s = self.by_id[r["meta"]["id"]]
            msgs, _ = initial_messages(rc, s)
            self.assertEqual(r["messages"][:2], msgs, s.id)

    def test_fewshot_app_filters(self):
        pool = FewShotPool.from_samples(self.samples, self.cat)
        s = self.by_id["train_0002"]                     # turns off a light: smart_home
        allowed = usable_apps(["contacts", "sms"], self.cat)
        self.assertIn("sys", allowed)
        got = pool.search(s.audio_text, 5, exclude={s.id}, allowed_apps=allowed)
        self.assertTrue(got and all(set(e["apps"]) <= allowed for e in got))
        self.assertIn("train_0004", [e["id"] for e in pool.search("nhắn tin cho anh minh", 5, allowed_apps=allowed)])
        other = pool.search(s.audio_text, 5, exclude={s.id}, avoid_apps={"smart_home"})
        self.assertTrue(other and all("smart_home" not in e["apps"] for e in other))
        self.assertIn("smart_home", pool.search(s.audio_text, 1, exclude={s.id})[0]["apps"])  # nearest is the twin

    def test_sft_and_inference_share_the_installed_filter(self):
        pool = FewShotPool.from_samples(self.samples, self.cat)
        pcfg = PromptConfig(fewshot_installed_only=True)
        rows, _ = build_rows(self.samples, self.cat, None, pool, pcfg, SFTConfig(n_aug=0))
        rc = RunContext(self.sp, self.cat, text_source="oracle", pool=pool, prompt_cfg=pcfg)
        for r in rows:
            sm = self.by_id[r["meta"]["id"]]
            msgs, _ = initial_messages(rc, sm)
            self.assertEqual(r["messages"][:2], msgs, sm.id)
        rows2, stat = build_rows(self.samples, self.cat, None, pool, PromptConfig(),
                                 SFTConfig(n_aug=0, p_fewshot_other_app=1.0))
        self.assertGreater(stat["aug:fewshot_other_app"], 0)
        for r in rows2:
            sm = self.by_id[r["meta"]["id"]]
            if "fewshot_other_app" not in r["meta"]["aug"]:
                continue
            own = {c["name"].split(".")[0] for c in sm.gold_calls} - {"sys"}
            fs = r["messages"][0]["content"].split("# Ví dụ đã làm đúng")[-1]
            for app in own:
                self.assertNotIn(f"⇒ {app}.", fs, sm.id)

    def test_late_steps_get_fewer_tokens(self):
        import time as _t
        from lifemate.agent import Agent, AgentConfig
        from lifemate.llm import ScriptedBackend

        class Rec(ScriptedBackend):
            def generate(self, reqs):
                self.sizes = getattr(self, "sizes", []) + [r.max_new_tokens for r in reqs]
                return super().generate(reqs)

        s = self.by_id["train_0002"]
        rc = RunContext(self.sp, self.cat, text_source="oracle", formats=self.fb)
        script = {s.id: ['{"type": "finish"}']}
        cfg = AgentConfig(deadline_s=25.0, hard_limit_s=28.0, min_tok_per_s=40.0, max_new_tokens=512)
        for age, want in ((1.0, 512), (20.0, 320), (27.5, None)):
            ep = build_episode(rc, s)
            ep.t0 = _t.time() - age
            b = Rec(script)
            Agent(b, cfg).run([ep])
            if want is None:                              # under 32 tokens left: stop, nothing generated
                self.assertEqual(ep.stop_reason, "deadline")
                self.assertFalse(getattr(b, "sizes", []))
            else:
                self.assertAlmostEqual(b.sizes[0], want, delta=8)
                self.assertEqual(ep.stop_reason, "finish")
        ep1, ep2 = build_episode(rc, s), build_episode(rc, self.by_id["train_0007"])
        ep1.t0 = ep2.t0 = _t.time() - 27.9                # batch runs have no per-sample clock
        b = Rec({s.id: ['{"type": "finish"}'], "train_0007": ['{"type": "finish"}']})
        Agent(b, cfg).run([ep1, ep2])
        self.assertEqual(b.sizes, [512, 512])

    def test_sft_trajectories_replay_in_env(self):
        rows, _ = build_rows(self.samples, self.cat, None, None, PromptConfig(), SFTConfig(n_aug=0))
        for r in rows:
            acts = [parse_action(m["content"]) for m in r["messages"] if m["role"] == "assistant"]
            s = self.by_id[r["meta"]["id"]]
            calls = [{"name": a["name"], "arguments": a["arguments"]} for a in acts if a["type"] != "finish"]
            self.assertEqual(calls, s.gold_calls, s.id)

    def test_augmented_rows_are_self_consistent(self):
        rows, stat = build_rows(self.samples, self.cat, None, None, PromptConfig(),
                                SFTConfig(n_aug=3, p_rename=1.0, p_distractor=0.5, p_remove_oos=0.5, seed=1))
        self.assertGreater(stat.get("aug:rename", 0), 0)
        for r in rows:
            system = r["messages"][0]["content"]
            for m in r["messages"]:
                if m["role"] != "assistant":
                    continue
                a = parse_action(m["content"])
                if a["type"] != "finish":
                    self.assertIn(f"- {a['name']} [", system, (r["meta"], a["name"]))
                    if a["type"] == "ask":
                        self.assertIn(f"- {a['arguments']['tool_name']} [", system)

    def test_leave_apps_out_no_leak(self):
        spl = leave_apps_out(self.samples, self.cat, n_apps=1, seed=0)
        held = set(spl["held_apps"])
        train = [s for s in self.samples if s.id in set(spl["train_ids"])]
        rows, _ = build_rows(train, self.cat, None, None, PromptConfig(),
                             SFTConfig(n_aug=2, p_distractor=1.0), held, set(spl["val_ids"]))
        for r in rows:
            for a in held:
                self.assertNotIn(f"## {a} ", r["messages"][0]["content"])

    def test_transcript_formats(self):
        from lifemate.data import Sample
        base = {"id": "x", "audio": "audio/x/x.wav",
                "history": [{"role": "user", "audio": "audio/x/x_u1.wav"},
                            {"role": "assistant", "text": "Bạn muốn chuyển bao nhiêu?"}]}
        forms = [
            {"audio_text": "hai trăm nghìn", "history": [{"role": "user", "audio": "audio/x/x_u1.wav",
                                                         "audio_text": "chuyển tiền cho mẹ"}, base["history"][1]]},
            {"audio_text": ["chuyển tiền cho mẹ", "hai trăm nghìn"]},
            {"audio_text": {"audio/x/x_u1.wav": "chuyển tiền cho mẹ", "audio/x/x.wav": "hai trăm nghìn"}},
            {"audio_text": {"final": "hai trăm nghìn", "history": ["chuyển tiền cho mẹ"]}},
        ]
        for f in forms:
            s = Sample({**base, **f})
            self.assertEqual(s.transcripts(), ("hai trăm nghìn", ["chuyển tiền cho mẹ"]), f)
        no_role = {"id": "y", "audio": "a.wav", "history": [{"audio": "u1.wav"}, {"text": "ok"}]}
        self.assertEqual(len(Sample(no_role).user_turns()), 1)

    def test_location_in_prompt(self):
        rc = RunContext(self.sp, self.cat, text_source="oracle")
        msgs, _ = initial_messages(rc, self.by_id["train_0001"])
        self.assertIn("Vị trí hiện tại", msgs[1]["content"])

    def test_asr_transcripts_used(self):
        tr = Transcripts()
        s = self.by_id["train_0006"]
        tr.rows[s.id] = {"id": s.id, "final": "hai trăm rưỡi ngàn", "history": ["chuyển tiền cho mẹ"]}
        rc = RunContext(self.sp, self.cat, transcripts=tr, text_source="asr")
        msgs, _ = initial_messages(rc, s)
        self.assertIn("hai trăm rưỡi ngàn", msgs[1]["content"])
        self.assertIn("số tiền 250000", msgs[1]["content"])
        ep = build_episode(rc, s)
        self.assertIn("c_me", ep.gstate.seen)


if __name__ == "__main__":
    unittest.main()
