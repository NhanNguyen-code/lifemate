"""Batched agent loop: action -> guard -> ToolEnv.call -> result back to the model."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from . import recovery
from .actions import KIND_TO_TYPE, TYPE_PREFIX, TYPES, action_text, canon_key, parse_action
from .guard import Guard, GuardState, app_of
from .llm import Backend, GenRequest, ScoreRequest
from .prompt import PromptConfig, assistant_message, compact_json, feedback_message, result_message


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_rejects_per_step: int = 2
    deadline_s: float = 25.0           # one-sample runs: no new step after this (ASR included)
    hard_limit_s: float = 28.0         # ... and no step may run past this: the rules cut at 30 s per sample
    min_tok_per_s: float = 40.0        # slowest decode speed assumed when sizing a late step (L4 ~40, A30 ~80)
    use_type_scoring: bool = False
    bias_first: dict = field(default_factory=dict)
    bias_later: dict = field(default_factory=dict)
    temperature: float = 0.0
    max_new_tokens: int = 512
    seed: int = 0
    force_ask_name: bool = True
    retry_idle_finish: bool = False    # a first-step finish whose thought says there is something to do gets 1 retry
    # the options below are off unless the config turns them on (lifemate/recovery.py says why they exist)
    recover: bool = False              # empty search / done=false: a note with what to change, no stop before a retry
    recover_nudges: int = 1            # finish or ask sent back at most this often per sample
    recover_lost: bool = False         # a finish after some call whose thought says information is missing: sent back
    defer_ask: bool = False            # an ask waits while the parts that need no answer are done; it is sent last
    defer_ask_steps: int = 4           # steps allowed after a deferred ask
    notice_repeat: bool = False        # a repeated identical read says so next to the cached result
    vote_k: int = 1                    # >1: that many candidates per step (1 greedy + sampled), most common action wins
    vote_temperature: float = 0.7
    vote_safe: bool = False            # the vote never turns an action into a stop (ask / finish), needs more than
                                       # half of the candidates to replace an action the guard would run, and puts
                                       # a read in place of one it would reject (Agent._safe_winner)
    vote_max_step: int = 0             # >0: candidates only on the first N steps (later steps: the usual single one)
    vote_until_s: float = 0.0          # one-sample runs: no candidates once the sample has used this many seconds
    recover_auto: bool = False         # a stop right after an empty search: the agent searches again itself, without
                                       # the word of address, then without filters (recovery.relaxed)
    recover_force: bool = False        # ... and where it has no such call, the model must write the call again
    recover_force_k: int = 5           # candidates for that call (1 greedy + sampled): the most common new one runs
    recover_calls: int = 3             # calls made this way per sample, at most
    consult: str = ""                  # "write" | "handover": at the first stop (finish / ask) with no write run yet,
                                       # the second adapter (another training recipe) is asked on the same history.
                                       # write: only a write of its that the guard runs is taken, then the main model
                                       # goes on; handover: a read too, and that model then ends the sample
    consult_until_s: float = 15.0      # one-sample runs: no consult once the sample has used this many seconds
    consult_self: bool = False         # no second adapter: the main model itself is asked, in the other prompt format
    force_lookup: bool = False         # the guard told the model which call to make (the read that finds an ID, or the
                                       # tool to call without the optional answer) and it asks again until the limit
                                       # or stops instead: the agent writes the start of that call, the model the
                                       # arguments (recover_force_k candidates, most common valid one); once per sample
    consult_thoughts: str = ""         # prompt format of the consult, "on" | "off" ("": the consult adapter's own,
                                       # or with consult_self the opposite of the main prompts)
    trace_results: int = 0             # characters of each tool result kept in the trace (0 = none)

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict | None) -> "AgentConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


# first-step finish with a thought like "đã đủ thông tin nên kết thúc" (seen on apps new to the model): the thought
# says there is work to do, the action does nothing
_WANTS = re.compile(r"đủ thông tin|xác định rõ|(cần|yêu cầu|muốn) (gửi|đặt|gọi|tìm|chuyển|tạo|bật|tắt|đổi|nạp|mua|"
                    r"thêm|kiểm tra|xem|hủy|huỷ|trả|dời|nhắn|ghi|lưu|kết thúc)", re.I)
_DONE = re.compile(r"không thể|ngoài (phạm vi|khả năng)|không có (ứng dụng|tool|công cụ|app)|không (ứng dụng|tool|"
                   r"công cụ|app) nào|vi phạm|từ chối|không hỗ trợ|không liên quan|lừa đảo|bảo mật|nguy hiểm|"
                   r"không phù hợp|không cần|không yêu cầu|hủy bỏ|huỷ bỏ|thôi|giữ nguyên|đã (thực hiện|hoàn tất|xong|"
                   r"làm|có)|trước đó", re.I)
IDLE_FINISH_FEEDBACK = ("Bạn chưa gọi tool nào nên yêu cầu chưa được thực hiện. Nếu một ứng dụng đã cài làm được "
                        "việc này, hãy gọi tool phù hợp ngay (thiếu thông tin bắt buộc thì hỏi lại bằng tool hệ "
                        "thống). Chỉ trả finish khi không ứng dụng nào làm được.")


def idle_finish(thought: str | None) -> bool:
    return bool(thought) and bool(_WANTS.search(thought)) and not _DONE.search(thought)


CONSULT_MODES = ("write", "handover")


class Cand(NamedTuple):
    """One candidate action of a vote. key: the action as far as the state goes; ok: the guard would not reject
    it; fresh: it would really be executed (not answered from the cache)."""
    key: str | None
    ok: bool
    typ: str | None
    fresh: bool


class Episode:
    def __init__(self, uid: str, sample, env, messages: list[dict], guard: Guard,
                 prompt_cfg: PromptConfig, seed: int = 0):
        self.uid = uid
        self.sample = sample
        self.env = env
        self.messages = messages
        self.guard = guard
        self.gstate: GuardState = guard.init_state(sample)
        self.prompt_cfg = prompt_cfg
        self.seed = seed
        self.step = 0
        self.rejects = 0
        self.done = False
        self.stop_reason = ""
        self.pending_type: str | None = None
        self.t0 = time.time()
        self.elapsed = 0.0
        self.events: list[dict] = []
        self.type_scores: list[dict] = []
        self.finish_retried = False
        self.n_messages_init = len(messages)
        self.fail: dict | None = None          # last call that found / changed nothing and was not tried again
        self.nudges = 0
        self.deferred_ask: dict | None = None  # ask held back until the rest is done
        self.ask_step = 0
        self.vote: dict | None = None
        self.forced: str | None = None         # start of the next action, written for the model (a search again)
        self.held: str | None = None           # the stop the model sent before that; it stands if nothing new comes
        self.recoveries = 0                    # calls made to make up for an empty search
        self.consult_init: list[dict] | None = None   # the first messages as the consult adapter's prompts look
        self.consulted = False
        self.consult_pending: dict | None = None       # a stop waiting for the consult adapter's answer
        self.wrote = False                     # a write ran, whatever it returned
        self.driver = "main"                   # "consult" after a handover: that adapter writes the next actions
        self.consult_thoughts: bool | None = None   # prompt format of the consult adapter (thoughts or not)
        self.named: str | None = None          # the call a guard rejection told the model to make, not made since
        self.named_forced = False              # force_lookup used on this sample
        self.force_name: str | None = None     # the forced start is that call (None: a search again, recover_force)
        self.force_stop: str | None = None     # how the sample stops if no valid call comes and no stop is held

    def consult_messages(self) -> list[dict]:
        """The history as the consult adapter sees it: its own first messages, then the same steps."""
        head = self.consult_init if self.consult_init is not None else self.messages[:self.n_messages_init]
        return head + self.messages[self.n_messages_init:]

    def stop(self, reason: str):
        if not self.done:
            self.done = True
            self.stop_reason = reason
            self.elapsed = time.time() - self.t0

    def prefix(self, cfg: AgentConfig) -> str:
        if self.pending_type is None:
            return ""
        ask = self.guard.ask
        if self.pending_type == "ask" and cfg.force_ask_name and ask is not None:
            return TYPE_PREFIX + f'ask", "name": "{ask.tool_name}", "arguments": {{'
        return TYPE_PREFIX + self.pending_type + '", "name": "'

    def result(self) -> dict:
        return {"id": self.sample.id, "uid": self.uid,
                "tool_calls": self.env.get_tool_calls(),
                "state_hash": self.env.get_state_hash(),
                "stop_reason": self.stop_reason, "elapsed": round(self.elapsed, 3),
                "steps": self.step, "events": self.events, "type_scores": self.type_scores}


class Agent:
    def __init__(self, backend: Backend, cfg: AgentConfig | None = None):
        self.backend = backend
        self.cfg = cfg or AgentConfig()
        self._single = False                   # run() on one sample: the time limits apply

    def _prompt(self, ep: Episode, consult: bool = False) -> str:
        if not self.backend.needs_prompt:
            return ""
        return self.backend.render(ep.consult_messages() if consult or ep.driver == "consult" else ep.messages)

    def _stop(self, e: Episode, reason: str):
        """Every stop goes through here: an ask that was held back is sent now, as the last call (gold order)."""
        if e.done:
            return
        if e.deferred_ask is not None:
            act, e.deferred_ask = e.deferred_ask, None
            try:
                result = e.env.call(act["name"], act["arguments"])
            except Exception as ex:
                result = {"ok": False, "error": f"lỗi môi trường: {ex}"}
            e.guard.record(act, result, e.gstate)
            e.events.append({"step": e.step, "kind": "call", "name": act["name"], "deferred": True,
                             "ok": bool(isinstance(result, dict) and result.get("ok"))})
            reason = "ask"
        e.stop(reason)

    def run(self, episodes: list[Episode]) -> list[Episode]:
        cfg = self.cfg
        # in a batch every step waits for the whole batch, so wall time says nothing about one sample;
        # the per-sample deadline is for batch-1 runs (predict.sh)
        deadline = cfg.deadline_s if len(episodes) == 1 else 0
        self._single = len(episodes) == 1
        active = [e for e in episodes if not e.done]
        while active:
            now = time.time()
            for e in active:
                if e.consult_pending is not None:
                    if (deadline and now - e.t0 > deadline) or e.step >= cfg.max_steps:
                        self._consult_skip(e, "time")
                elif deadline and now - e.t0 > deadline:
                    self._stop(e, "deadline")
                elif e.step >= cfg.max_steps:
                    self._stop(e, "max_steps")
                elif e.deferred_ask is not None and e.step - e.ask_step > cfg.defer_ask_steps:
                    self._stop(e, "ask")
            active = [e for e in active if not e.done]
            if not active:
                break
            # optional: pick the action type from biased log-probs, then force it as a prefix
            if cfg.use_type_scoring:
                # a forced type prefix would skip the thought, so models trained with thoughts generate freely
                need = [e for e in active if e.pending_type is None and not e.prompt_cfg.thoughts and not e.forced
                        and e.consult_pending is None and e.driver == "main"]
                if need:
                    reqs = [ScoreRequest(e, self._prompt(e) + TYPE_PREFIX, [t + '"' for t in TYPES]) for e in need]
                    for e, sc in zip(need, self.backend.score(reqs)):
                        bias = cfg.bias_first if e.step == 0 else cfg.bias_later
                        adj = {t: s + float(bias.get(t, 0.0)) for t, s in zip(TYPES, sc)}
                        choice = max(TYPES, key=lambda t: (adj[t], -TYPES.index(t)))
                        e.type_scores.append({"step": e.step, "raw": dict(zip(TYPES, sc)), "choice": choice})
                        if choice == "finish":
                            self._stop(e, "finish")
                        else:
                            e.pending_type = choice
                    active = [e for e in active if not e.done]
                    if not active:
                        break
            reqs, owners = [], []
            for e in active:
                max_new = cfg.max_new_tokens
                used = time.time() - e.t0
                if deadline:                   # a late step gets fewer tokens so it ends before the hard limit
                    budget = int((cfg.hard_limit_s - used) * cfg.min_tok_per_s)
                    if budget < 32:
                        if e.consult_pending is not None:
                            self._consult_skip(e, "time")
                        else:
                            self._stop(e, "deadline")
                        continue
                    max_new = min(max_new, budget)
                seed = e.seed * 1000 + e.step * 10 + e.rejects
                if e.consult_pending is not None:    # one greedy answer of the consult adapter, nothing forced
                    reqs.append(GenRequest(e, self._prompt(e, consult=True), "", 0.0, max_new, seed=seed,
                                           adapter="consult"))
                    owners.append((e, 1, "consult"))
                    continue
                other = e.driver == "consult"
                forced = bool(e.forced)
                pre = e.forced or e.prefix(cfg)
                prompt = (self._prompt(e) + pre) if self.backend.needs_prompt else ""
                if forced:
                    k = max(1, cfg.recover_force_k)
                elif (cfg.vote_k > 1 and not pre and not other
                      and not (cfg.vote_max_step and e.step >= cfg.vote_max_step)
                      and not (deadline and cfg.vote_until_s and used > cfg.vote_until_s)):
                    k = cfg.vote_k
                else:
                    k = 1
                for j in range(k):             # candidate 0 is the usual (greedy) one, the others are sampled
                    reqs.append(GenRequest(e, prompt, pre, cfg.temperature if j == 0 else cfg.vote_temperature,
                                           max_new, seed=seed + 100003 * j, adapter="consult" if other else None))
                owners.append((e, k, "forced" if forced else ""))
            if not reqs:
                break
            outs = self.backend.generate(reqs)
            i = 0
            for e, k, kind in owners:
                texts = [reqs[i + j].prefix + outs[i + j] for j in range(k)]
                i += k
                if kind == "consult":
                    self._consult_done(e, texts[0])
                elif kind == "forced":
                    self._handle_forced(e, texts)
                else:
                    self._handle(e, self._vote(e, texts) if k > 1 else texts[0])
            active = [e for e in active if not e.done]
        for e in episodes:
            if not e.done:
                self._stop(e, "done")
        return episodes

    def _look(self, e: Episode, a: dict):
        """guard.check without using up a one-time warning: for candidates that may never be run."""
        warned = set(e.gstate.warned)
        d = e.guard.check(a, e.env, e.gstate)
        e.gstate.warned = warned
        return d

    def _vote_key(self, e: Episode, a: dict | None) -> Cand:
        """Same key = same action as far as the state goes: the guard's fixes are applied first and free-text
        arguments do not count."""
        if a is None:
            return Cand(None, False, None, False)
        if a["type"] == "finish":
            return Cand("finish", True, "finish", True)
        d = self._look(e, a)
        ok = d.kind != "reject"
        act = d.action if ok and d.action.get("name") else a
        name = str(act.get("name"))
        spec = e.guard.cat.tools.get(name)
        args = act.get("arguments") or {}
        if spec is not None and spec.params:
            args = {k: v for k, v in args.items() if k in spec.params and not spec.params[k].fuzzy}
        typ = KIND_TO_TYPE.get(spec.kind, "write") if spec is not None else a.get("type")
        return Cand(canon_key(name, args), ok, typ, d.kind == "execute")

    @staticmethod
    def _safe_winner(cands: list[Cand]) -> Cand:
        """What a vote may change. A wrong write or a needless question costs the whole sample and a read costs
        little, and a stop has few ways to be written, so it gathers sampled votes too easily:
        - the greedy action stays unless more than half of the candidates agree on another one, and never gives way
          to a stop of another kind (a call turned into ask / finish, an ask into finish);
        - a greedy action the guard would reject gives way to such a majority, or else to the most common read that
          would really run; with neither it is rejected as usual and the model gets the feedback."""
        g, n = cands[0], len(cands)
        run = [c for c in cands if c.ok]
        counts = Counter(c.key for c in run)

        def best(pred):
            opts = [c for c in run if pred(c)]
            if not opts:
                return None, 0
            top = max(counts[c.key] for c in opts)
            return next(c for c in opts if counts[c.key] == top), top

        if g.ok:
            w, top = best(lambda c: True)
            if counts[g.key] == top:
                return g
            return w if top * 2 > n and (w.typ in ("read", "write") or w.typ == g.typ) else g
        w, top = best(lambda c: c.typ in ("read", "write"))
        if w is not None and top * 2 > n:
            return w
        w, _ = best(lambda c: c.typ == "read" and c.fresh)
        return w if w is not None else g

    def _vote(self, e: Episode, texts: list[str]) -> str:
        cands = [self._vote_key(e, parse_action(t)) for t in texts]
        pool = [c for c in cands if c.ok] or [c for c in cands if c.key is not None]    # runnable ones first
        if not pool:
            return texts[0]
        counts: dict[str, int] = {}
        for c in pool:
            counts[c.key] = counts.get(c.key, 0) + 1
        top = max(counts.values())
        # ties go to the greedy candidate, then to the earliest one
        lead = cands[0] if cands[0] in pool and counts[cands[0].key] == top else next(
            c for c in pool if counts[c.key] == top)
        win = self._safe_winner(cands) if self.cfg.vote_safe else lead
        e.vote = {"k": len(texts), "top": counts.get(win.key, 0), "greedy": cands[0] == win}
        if win != lead:                            # what the plain count would have picked instead
            e.vote["over"] = lead.typ
        if len(set(cands)) > 1:                    # the split, greedy first: [type, candidates, guard would run it]
            tally = Counter(cands)
            e.vote["split"] = [[c.typ, n, int(c.ok)] for c, n in tally.items()]
        return texts[cands.index(win)]

    def _reject(self, e: Episode, raw_action: dict | None, raw_text: str, msg: str):
        e.rejects += 1
        e.events.append({"step": e.step, "kind": "reject", "text": raw_text[:500], "error": msg})
        e.vote = None
        if e.rejects > self.cfg.max_rejects_per_step:
            if not self._force_named(e, None, "rejected"):
                self._stop(e, "rejected")
            return
        content = action_text(raw_action) if raw_action and raw_action.get("type") and raw_action.get(
            "name") else raw_text.strip()[:500]
        e.messages.append({"role": "assistant", "content": content})
        e.messages.append(feedback_message(msg))
        e.pending_type = None

    def _search_again(self, e: Episode) -> bool:
        """The agent's own calls after an empty search (recovery.relaxed), until one finds something.
        True: there is a new result with data in the history, the model goes on from it."""
        cfg = self.cfg
        while e.fail and e.fail["kind"] == "empty" and e.recoveries < cfg.recover_calls:
            f = e.fail
            spec = e.guard.cat.tools.get(f["name"])
            nxt = None
            for args, how in recovery.relaxed(spec, f["args"]):
                d = self._look(e, {"type": "read", "name": f["name"], "arguments": args})
                if d.kind == "execute":            # not tried before (that would be "cached") and valid
                    nxt = (d.action, how)
                    break
            if nxt is None:
                return False
            act, how = nxt
            e.recoveries += 1
            thought = recovery.auto_thought(how, act["name"]) if e.prompt_cfg.thoughts else None
            if self._execute(e, act, thought, {}, auto=how) is None:
                return True
        return False

    def _hold(self, e: Episode, a: dict, text: str) -> bool:
        """A stop (finish / ask) right after a call that found or changed nothing does not stand yet.
        Empty search: the agent searches again itself, or has the model write the call again (recover_auto /
        recover_force). Otherwise, and for a write that did nothing: sent back once with a note."""
        cfg, f = self.cfg, e.fail
        if not (cfg.recover and f):
            return False
        if f["kind"] == "empty" and (cfg.recover_auto or cfg.recover_force) and e.recoveries < cfg.recover_calls:
            if cfg.recover_auto and self._search_again(e):
                return True
            f = e.fail
            if (cfg.recover_force and f and f["kind"] == "empty" and f["args"] and not f.get("forced")
                    and e.recoveries < cfg.recover_calls):
                spec = e.guard.cat.tools.get(f["name"])
                thought = (recovery.retry_thought(spec, f["args"], e.sample.context, e.guard.text)
                           if e.prompt_cfg.thoughts else None)
                e.forced, e.held = recovery.retry_prefix(thought, "read", f["name"]), text
                f["forced"] = f["retried"] = True
                e.recoveries += 1
                e.events.append({"step": e.step, "kind": "hold", "name": f["name"], "stop": a.get("type")})
                return True
            if f is None:
                return False
        if f.get("retried") or e.nudges >= cfg.recover_nudges:
            return False
        e.nudges += 1
        self._reject(e, a, text, recovery.nudge(f))
        return True

    def _handle_forced(self, e: Episode, texts: list[str]):
        """The model wrote the failed search again, or the call the guard named (k candidates). The most common call
        that is new and valid runs; with none, the stop the model had sent stands."""
        held, e.forced, e.held = e.held, None, None
        named, stop = e.force_name, e.force_stop
        e.force_name = e.force_stop = None
        name = named or (e.fail or {}).get("name")
        seen: dict[str, list] = {}
        for t in texts:
            a = parse_action(t)
            if a is None or a.get("type") == "finish":
                continue
            d = self._look(e, a)
            if d.kind != "execute" or d.action["name"] != name:
                continue                           # the same call again ("cached"), not valid, or another tool
            seen.setdefault(canon_key(name, d.action["arguments"]), [0, t])[0] += 1
        if not seen:
            e.events.append({"step": e.step, "kind": "lookup_none" if named else "retry_none", "name": name})
            if not named:
                e.recoveries = self.cfg.recover_calls
            if held:
                self._handle(e, held)
            else:
                self._stop(e, stop or "finish")
            return
        top = max(c for c, _ in seen.values())
        text = next(t for c, t in seen.values() if c == top)       # order of appearance: the greedy one first
        self._handle(e, text, {"lookup": name} if named else {"forced": True})

    def _thoughts(self, e: Episode) -> bool:
        """Whether the model writing the next action uses the prompt format with thoughts."""
        if e.driver == "consult" and e.consult_thoughts is not None:
            return e.consult_thoughts
        return e.prompt_cfg.thoughts

    def _force_named(self, e: Episode, held: str | None, why: str) -> bool:
        """force_lookup: the call a guard rejection named (e.named) is started for the model, which writes the
        arguments. held: the stop the model sent instead, carried out if no valid call comes (None: stop with why)."""
        name = e.named
        if (not self.cfg.force_lookup or not name or e.named_forced or name not in e.guard.allowed_set
                or e.consult_pending is not None):
            return False
        typ = KIND_TO_TYPE.get(e.guard.cat.kind(name) or "", "write")
        if typ not in ("read", "write"):
            return False
        e.named_forced = True
        thought = recovery.named_thought(name, typ) if self._thoughts(e) else None
        e.forced, e.held = recovery.retry_prefix(thought, typ, name), held
        e.force_name, e.force_stop = name, why
        e.rejects = 0
        e.pending_type = None
        e.events.append({"step": e.step, "kind": "lookup_hold", "name": name, "stop": why})
        return True

    @staticmethod
    def _made(e: Episode, act: dict):
        """A call ran (or came from the cache): the call the guard had named is no longer missing once it, a read of
        the same app (for a named read) or any write has run."""
        n = e.named
        if n and (act["name"] == n or act["type"] == "write"
                  or (act["type"] == "read" and app_of(act["name"]) == app_of(n) and e.guard.cat.kind(n) == "read")):
            e.named = None

    def _execute(self, e: Episode, act: dict, thought: str | None, note: dict, auto: str | None = None):
        """env.call, the trace event, and the action with its result in the history. Returns recovery.outcome.
        auto: the call is the agent's own ("title" / "all"), not one the model wrote."""
        cfg = self.cfg
        try:
            result: Any = e.env.call(act["name"], act["arguments"])
        except Exception as ex:  # one env error must not kill the whole split
            result = {"ok": False, "error": f"lỗi môi trường: {ex}"}
        if act["type"] == "write":
            e.wrote = True
        self._made(e, act)
        e.guard.record(act, result, e.gstate)
        # zaic_tools answers an unknown ID, a short balance... with ok=true, done=false (state unchanged)
        out = recovery.outcome(act["type"], act["arguments"], result)
        ev = {"step": e.step, "kind": "call", "name": act["name"],
              "ok": bool(isinstance(result, dict) and result.get("ok")),
              **({"noop": True} if out == "noop" else {}), **({"empty": True} if out == "empty" else {}),
              **({"auto": auto} if auto else {}), **note}
        if cfg.trace_results:
            shown = compact_json(result, e.prompt_cfg.result_max_chars, e.prompt_cfg.list_max_items)
            ev["res"] = shown[:cfg.trace_results]
            if len(shown) < len(compact_json(result, 10 ** 9, 10 ** 9)):
                ev["cut"] = True
        e.events.append(ev)
        extra = ""
        if cfg.recover:
            app, prev = app_of(act["name"]), e.fail
            if out:
                same = bool(prev and prev.get("app") == app)
                e.fail = {"name": act["name"], "app": app, "kind": out, "type": act["type"],
                          "args": dict(act["arguments"]), "retried": same or bool(auto),
                          "forced": bool(same and prev.get("forced"))}
                if auto:
                    extra = recovery.auto_note(auto, False)
                else:
                    spec = e.guard.cat.tools.get(act["name"])
                    readers = [e.guard.cat.tools[t] for t in e.guard.readers(app)]
                    extra = recovery.hint(out, spec, act["arguments"], e.sample.context, readers,
                                          str(result.get("message") or ""))
            else:
                if prev and (act["type"] == "write" or (prev.get("app") == app and prev["type"] == "read")):
                    e.fail = None              # the retry worked, or the model went on to act; a write that did
                                               # nothing is only made good by a write that does something
                if auto:
                    extra = recovery.auto_note(auto, True)
        # the history keeps the thought as written, next to the action the guard actually ran
        e.messages.append(assistant_message({"thought": thought, **act} if thought else act))
        e.messages.append(result_message(act["name"], result, e.prompt_cfg, extra))
        e.step += 1
        e.rejects = 0
        e.pending_type = None
        return out

    def _handle(self, e: Episode, text: str, tag: dict | None = None):
        cfg = self.cfg
        a = parse_action(text)
        if a is None:
            self._reject(e, None, text, "không đọc được JSON hành động")
            return
        thought = a.get("thought")
        note = {"thought": thought[:300]} if thought else {}
        if tag:
            note.update(tag)
        if e.driver == "consult":
            note["by"] = "consult"
        if e.vote:
            note["vote"], e.vote = e.vote, None
        if (a["type"] == "finish" and cfg.retry_idle_finish and e.step == 0 and not e.finish_retried
                and idle_finish(thought)):
            e.finish_retried = True
            self._reject(e, a, text, IDLE_FINISH_FEEDBACK)
            return
        if a["type"] == "finish":
            if self._hold(e, a, text):
                return
            if self._force_named(e, text, "finish"):     # the call the guard named was never made
                return
            # with no call made yet this may be a request no app can serve: that finish is left alone
            if (cfg.recover_lost and e.step >= 1 and e.deferred_ask is None and e.nudges < cfg.recover_nudges
                    and recovery.lost(thought)):
                e.nudges += 1
                self._reject(e, a, text, recovery.LOST_FEEDBACK)
                return

            def finish_now():
                e.events.append({"step": e.step, "kind": "finish", **note})
                self._stop(e, "finish")
            if not self._consult_first(e, "finish", finish_now):
                finish_now()
            return
        d = e.guard.check(a, e.env, e.gstate)
        if d.kind == "reject":
            if d.call:
                e.named = d.call
            self._reject(e, a if a.get("type") else {**a, "type": d.action.get("type")}, text, d.message)
            return
        act = d.action
        if d.fixes:
            note["swap"] = d.fixes[0]
        if act["type"] == "ask":
            if e.guard.ask is not None and e.guard.ask.parse(a.get("arguments") or {})[1] != e.guard.ask.parse(
                    act["arguments"])[1]:
                note["ask_fixed"] = True        # the guard renamed or reordered what is asked for
            if e.deferred_ask is not None:      # one question per request: the first one stands
                e.events.append({"step": e.step, "kind": "drop", "name": act["name"], "why": "second ask", **note})
                self._stop(e, "ask")
                return
            if self._hold(e, {**a, "type": "ask"}, text):
                return
            if cfg.defer_ask and d.kind == "execute":
                tool = e.guard.ask.parse(act["arguments"])[0] if e.guard.ask is not None else None
                e.deferred_ask, e.ask_step = act, e.step
                e.events.append({"step": e.step, "kind": "ask_deferred", "name": act["name"], "tool": tool, **note})
                e.messages.append(assistant_message({"thought": thought, **act} if thought else act))
                e.messages.append(result_message(act["name"], {"ok": True}, e.prompt_cfg,
                                                 recovery.ask_deferred_note(tool)))
                e.step += 1
                e.rejects = 0
                e.pending_type = None
                return
            if self._consult_first(e, "ask", lambda: self._run_action(e, d, act, thought, note)):
                return
        elif e.deferred_ask is not None and e.guard.ask is not None:
            # the tool the question is about cannot run before the answer: the model is done
            tool = e.guard._fix_name(str(e.guard.ask.parse(e.deferred_ask["arguments"])[0] or ""))
            if tool == act["name"]:
                e.events.append({"step": e.step, "kind": "drop", "name": act["name"], "why": "asked tool", **note})
                self._stop(e, "ask")
                return
        self._run_action(e, d, act, thought, note)

    def _run_action(self, e: Episode, d, act: dict, thought: str | None, note: dict):
        """An action the guard let through: run on the env (or its cached result back); an ask ends the sample."""
        if d.kind == "cached":
            self._made(e, act)
            e.events.append({"step": e.step, "kind": "cached", "name": act["name"], "why": d.message, **note})
            extra = recovery.REPEAT_NOTE if self.cfg.notice_repeat and d.message == "repeated read" else ""
            e.messages.append(assistant_message({"thought": thought, **act} if thought else act))
            e.messages.append(result_message(act["name"], d.result, e.prompt_cfg, extra))
            e.step += 1
            e.rejects = 0
            e.pending_type = None
        else:
            self._execute(e, act, thought, note)
        if act["type"] == "ask":   # gold always stops right after an ask
            self._stop(e, "ask")

    # consult: a second adapter looks at the first stop without a write (AgentConfig.consult). On the leave-apps-out
    # val the two recipes end without a write on different samples; where only the main one gives up, the other one's
    # write is right far more often than not (val: 18 better, 2 worse out of 42 such samples)

    def _consult_first(self, e: Episode, on: str, resume) -> bool:
        """True: the stop waits for the consult adapter (asked in the next batch of run()); resume() carries the
        stop out unchanged if nothing of its answer is taken."""
        cfg = self.cfg
        if (cfg.consult not in CONSULT_MODES or e.consulted or e.wrote or e.driver != "main"
                or not self.backend.has_adapter("consult")):
            return False
        if self._single and cfg.consult_until_s and time.time() - e.t0 > cfg.consult_until_s:
            return False
        e.consulted = True
        e.consult_pending = {"on": on, "resume": resume}
        return True

    def _consult_skip(self, e: Episode, why: str):
        pend, e.consult_pending = e.consult_pending, None
        e.events.append({"step": e.step, "kind": "consult", "on": pend["on"], "took": False, "why": why})
        pend["resume"]()

    def _consult_done(self, e: Episode, text: str):
        """The consult adapter's answer at the stop. Taken if the guard would run it as it is and it is a write
        (handover: or a read); otherwise the stop stands. One look: nothing is sent back to it."""
        cfg = self.cfg
        pend = e.consult_pending
        a = parse_action(text)
        take = ("write",) if cfg.consult == "write" else ("read", "write")
        d = self._look(e, a) if a is not None and a.get("type") in ("read", "write") and a.get("name") else None
        thought = (a or {}).get("thought")
        if d is None or d.kind != "execute" or d.action.get("type") not in take:
            got = d.action if d is not None and d.action.get("name") else (a or {})
            e.consult_pending = None
            e.events.append({"step": e.step, "kind": "consult", "on": pend["on"], "took": False,
                             "type": got.get("type"), "name": got.get("name"),
                             **({"thought": thought[:300]} if thought else {})})
            pend["resume"]()
            return
        d = e.guard.check(a, e.env, e.gstate)          # for real now: one-time warnings are used up as usual
        if d.kind != "execute":
            self._consult_skip(e, "guard")
            return
        e.consult_pending = None
        if cfg.consult == "handover":
            e.driver = "consult"
        note = {**({"thought": thought[:300]} if thought else {}), "consult": {"on": pend["on"], "mode": cfg.consult},
                **({"swap": d.fixes[0]} if d.fixes else {})}
        # after a write the main model goes on: the history stays in its format (no thought)
        self._execute(e, d.action, thought if e.driver == "consult" else None, note)
