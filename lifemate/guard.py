"""Checks run before every env.call. Every executed call ends up in the submission, so fix or reject first."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .actions import KIND_TO_TYPE, canon_key
from .catalog import Catalog, ParamSpec
from .textnorm import strip_diacritics
from .timeutil import DATETIME_SHAPES, FormatBook, coerce_shape, shape_of


@dataclass
class GuardConfig:
    check_xref: bool = True
    cache_reads: bool = True
    block_duplicate_writes: bool = True
    drop_unknown_args: bool = True
    coerce_types: bool = True
    fix_time_formats: bool = True
    ask_param_order: str = "keep"
    # the optional rules below are only switched on where the train labels never break them
    cache_history_reads: bool = False      # a read repeated from the history gets the history result, no call
    ask_needed_only: bool = False          # reject an ask that lists only optional params
    ask_ids_after_read: bool = False       # reject asking for an ID before any read of the app that holds it
    xref_typed: bool = False               # an ID must come from data of its own app (chat_id from chat_ott, ...)
    fix_names: bool = False                # "mẹ" -> "Mẹ", "Mai" -> "Cô giáo Mai": names as stored, seen in train
    drop_unused_params: bool = False       # a tool seen often in train: drop args gold never passes to it
    no_text_ask: bool = False              # an ask for free text only (content, title...) is sent back once: gold
                                           # never asks it, the text is in what the user said
    drop_optional_text_unseen: bool = False   # a write tool not in train: drop optional free-text args (gold: 15%)
    block_failed_repeats: bool = False     # a write that did nothing (done=false) is not sent again unchanged
    no_ask_single_hit: bool = False        # asking for an ID when the last read showed exactly one item is sent back
                                           # once (gold asks after 2+ hits)
    fix_ask_params: bool = False           # an ask that names a parameter the tool does not have: a field of a list
                                           # item becomes that list ("quantity" -> "items"), an ID becomes the one ID
                                           # the tool needs ("contact_id" -> "chat_id"); gold asks never need this
    prefer_ref_id: bool = False            # x-one-of [ID] | [raw value]: a raw value (a phone number) that a record
                                           # read earlier holds becomes that record's ID (train: 122 of 125 calls)

    @classmethod
    def from_json(cls, d: dict | None) -> "GuardConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


@dataclass
class GuardState:
    seen: set[str] = field(default_factory=set)
    read_cache: dict[str, Any] = field(default_factory=dict)
    writes_done: set[str] = field(default_factory=set)
    history_cache: dict[str, Any] = field(default_factory=dict)
    read_apps: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    hits: dict[str, int] = field(default_factory=dict)      # "app.collection" -> items in the last read of this turn
    warned: set[str] = field(default_factory=set)           # one-time ask rules already used on this sample
    seen_by_app: dict[str, set[str]] = field(default_factory=dict)
    seen_by_key: dict[str, set[str]] = field(default_factory=dict)
    rec_ids: dict[str, dict[str, str | None]] = field(default_factory=dict)  # app -> value -> ID of the record


@dataclass
class Decision:
    kind: str
    action: dict
    message: str = ""
    result: Any = None
    call: str | None = None                 # reject: the tool the message tells the model to call
    fixes: list[str] = field(default_factory=list)   # changes worth a mark in the trace ("phone_number->contact_id")


def collect_scalars(obj: Any, out: set[str], max_len: int = 120):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_scalars(v, out, max_len)
    elif isinstance(obj, list):
        for v in obj:
            collect_scalars(v, out, max_len)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, str)):
        s = str(obj)
        if 0 < len(s) <= max_len:
            out.add(s)


def _fold(s: str) -> str:
    return strip_diacritics(str(s)).lower().strip()


def nfc(v: Any) -> Any:
    # gold strings are NFC and non-fuzzy args are compared byte for byte
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if isinstance(v, list):
        return [nfc(x) for x in v]
    return v


# free text the model writes itself from what the user said (place strings can be asked for, gold does)
TEXT_PARAMS = {"body", "content", "label", "note", "subject", "text", "title", "message", "description"}
CREATES = ("create", "add", "new", "save")


def app_of(name: str) -> str:
    return str(name).split(".", 1)[0]


def needed_params(spec) -> set[str]:
    return set(spec.required) | {p for g in spec.one_of for p in g}


def _item_fields(p: ParamSpec) -> set[str]:
    items = p.raw.get("items")
    props = (items.get("properties") if isinstance(items, dict) else None) or p.raw.get("properties") or {}
    return set(props) if isinstance(props, dict) else set()


def ask_param(spec, name: str) -> tuple[str, str] | None:
    """The parameter of `spec` an ask meant by a name the tool does not have, when only one fits:
    (parameter, "inside") for a field of its list items, (parameter, "id") for the one ID the tool needs."""
    inside = [k for k, p in spec.params.items() if name in _item_fields(p)]
    if len(inside) == 1:
        return inside[0], "inside"
    if name == "id" or name.endswith(("_id", "_ids")):
        need = needed_params(spec)
        ids = [k for k, p in spec.params.items() if p.ref and k in need]
        if len(ids) == 1:
            return ids[0], "id"
    return None


def history_calls(history: list[dict]) -> list[tuple[dict, Any]]:
    out = []
    for turn in history:
        res = turn.get("tool_results") or []
        for i, c in enumerate(turn.get("tool_calls") or []):
            out.append((c, res[i] if i < len(res) else None))
    return out


def history_reads(catalog: Catalog, history: list[dict]) -> dict[str, Any]:
    """canon key -> result of each history read that no later history write to the same app made stale."""
    calls = history_calls(history)
    out = {}
    for i, (c, r) in enumerate(calls):
        name = str(c.get("name") or "")
        if catalog.kind(name) != "read" or r is None or (isinstance(r, dict) and r.get("ok") is False):
            continue
        later = calls[i + 1:]
        if any(catalog.kind(str(c2.get("name") or "")) == "write" and app_of(c2.get("name") or "") == app_of(name)
               for c2, _ in later):
            continue
        out[canon_key(name, c.get("arguments") or {})] = r
    return out


def read_apps_in_history(catalog: Catalog, history: list[dict]) -> set[str]:
    return {app_of(c.get("name") or "") for c, r in history_calls(history)
            if catalog.kind(str(c.get("name") or "")) == "read" and not (isinstance(r, dict) and r.get("ok") is False)}


def collect_keyed(obj: Any, out: dict[str, set[str]], key: str | None = None, max_len: int = 120):
    if isinstance(obj, dict):
        for k, v in obj.items():
            collect_keyed(v, out, str(k), max_len)
    elif isinstance(obj, list):
        for x in obj:
            collect_keyed(x, out, key, max_len)
    elif key is not None and isinstance(obj, (str, int, float)) and not isinstance(obj, bool):
        if len(str(obj)) <= max_len:
            out.setdefault(key, set()).add(str(obj))


def note_hits(catalog: Catalog, hits: dict[str, int], name: str, result: Any):
    """How many items each list of a read result holds: contacts.search -> {"contacts.contacts": 2}."""
    if catalog.kind(name) != "read" or not isinstance(result, dict):
        return
    for k, v in result.items():
        if isinstance(v, list):
            hits[f"{app_of(name)}.{k}"] = len(v)


def note_call(catalog: Catalog, by_app: dict[str, set[str]], name: str, args: dict | None, result: Any,
              by_key: dict[str, set[str]] | None = None):
    """File every value of a call under the app it belongs to (results and plain args under the called app, an
    x-ref arg under the app its ref points to) and under the key it came with (contact_id in a call log)."""
    app = app_of(name)
    spec = catalog.tools.get(name)
    refs = spec.ref_keys() if spec else {}
    for k, v in (args or {}).items():
        collect_scalars(v, by_app.setdefault(app_of(refs[k]) if k in refs else app, set()))
    if result is not None:
        collect_scalars(result, by_app.setdefault(app, set()))
    if by_key is not None:
        collect_keyed(args or {}, by_key)
        collect_keyed(result, by_key)


def note_records(app: str, obj: Any, out: dict[str, dict[str, str | None]], max_len: int = 120):
    """Each scalar of a record that has an "id", filed under the app that returned it: value -> that ID
    (None once records with different IDs share the value)."""
    if isinstance(obj, list):
        for x in obj:
            note_records(app, x, out, max_len)
        return
    if not isinstance(obj, dict):
        return
    rid = obj.get("id")
    if isinstance(rid, (str, int)) and not isinstance(rid, bool) and str(rid):
        rid, m = str(rid), out.setdefault(app, {})
        for k, v in obj.items():
            if k == "id" or isinstance(v, bool) or not isinstance(v, (str, int)):
                continue
            s = str(v)
            if s and len(s) <= max_len:
                m[s] = rid if m.get(s, rid) == rid else None
    for v in obj.values():
        if isinstance(v, (dict, list)):
            note_records(app, v, out, max_len)


TITLES = {"anh", "chi", "em", "co", "chu", "bac", "ong", "ba", "cau", "mo", "di", "duong", "thim", "thay", "giao",
          "sep", "ban", "be", "chau", "con", "a"}


def canonical_name(v: str, names: set[str], text: str) -> str:
    """The stored form of a name the model wrote, when exactly one stored name fits."""
    if not names or not isinstance(v, str) or v in names:
        return v
    fv, ft = _fold(v), f" {_fold(text)} "
    same = [n for n in names if _fold(n) == fv]
    if len(same) == 1:
        return same[0]                       # case or tone marks only
    # title left out: "Mai" for "Cô giáo Mai", only if the user said the full stored name
    longer = [n for n in names if _fold(n).endswith(" " + fv) and f" {_fold(n)} " in ft]
    if len(longer) == 1:
        return longer[0]
    # title added: "anh Tuấn" when the contact is stored as "Tuấn"
    words = fv.split()
    shorter = [n for n in names for k in (1, 2)
               if len(words) > k and " ".join(words[k:]) == _fold(n) and set(words[:k]) <= TITLES]
    if len(set(shorter)) == 1:
        return shorter[0]
    return v


def coerce_value(v: Any, p: ParamSpec) -> Any:
    t = p.type
    if v is None:
        return v
    if t == "integer" and not isinstance(v, bool):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s):
                s = re.sub(r"[.,]", "", s)
            if re.fullmatch(r"-?\d+", s):
                return int(s)
    elif t == "number" and not isinstance(v, bool):
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            try:
                f = float(s)
                return int(f) if f.is_integer() and "." not in v else f
            except ValueError:
                return v
    elif t == "boolean":
        if isinstance(v, str):
            lv = _fold(v)
            if lv in ("true", "co", "bat", "yes", "1"):
                return True
            if lv in ("false", "khong", "tat", "no", "0"):
                return False
    elif t == "string":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
    elif t == "array":
        if not isinstance(v, list):
            return [v]
    if p.enum and v not in p.enum:
        fv = _fold(v) if isinstance(v, str) else None
        for e in p.enum:
            if isinstance(e, str) and fv is not None and _fold(e) == fv:
                return e
    return v


class Guard:
    def __init__(self, catalog: Catalog, allowed: list[str], ctx: dict,
                 cfg: GuardConfig | None = None, formats: FormatBook | None = None, clock: list[str] | None = None,
                 text: str = ""):
        self.cat = catalog
        self.clock = list(clock or [])         # "HH:MM" said by the user, in order
        self.text = text                       # everything the user said in this sample
        self.allowed = list(allowed)
        self.allowed_set = set(allowed)
        self.ctx = ctx
        self.cfg = cfg or GuardConfig()
        self.formats = formats
        self.ask = catalog.ask_format()

    def init_state(self, sample) -> GuardState:
        st = GuardState()
        for turn in sample.history:
            for c in turn.get("tool_calls") or []:
                collect_scalars(c.get("arguments") or {}, st.seen)
            for r in turn.get("tool_results") or []:
                collect_scalars(r, st.seen)
        st.read_apps = read_apps_in_history(self.cat, sample.history)
        for c, r in history_calls(sample.history):
            note_call(self.cat, st.seen_by_app, str(c.get("name") or ""), c.get("arguments"), r, st.seen_by_key)
            note_records(app_of(str(c.get("name") or "")), r, st.rec_ids)
        if self.cfg.cache_history_reads:
            st.history_cache = history_reads(self.cat, sample.history)
        return st

    def readers(self, app: str) -> list[str]:
        return [t for t in self.allowed if app_of(t) == app and self.cat.kind(t) == "read"]

    def finder(self, readers: list[str]) -> str | None:
        """The read to look an ID up with: the first one that needs no argument, else the first one."""
        free = [t for t in readers if not self.cat.tools[t].required and not self.cat.tools[t].one_of]
        return (free or readers or [None])[0]

    def record(self, action: dict, result: Any, st: GuardState):
        collect_scalars(result, st.seen)
        collect_scalars(action.get("arguments") or {}, st.seen)
        note_call(self.cat, st.seen_by_app, action["name"], action.get("arguments"), result, st.seen_by_key)
        note_records(app_of(action["name"]), result, st.rec_ids)
        note_hits(self.cat, st.hits, action["name"], result)
        key = canon_key(action["name"], action.get("arguments") or {})
        if action["type"] == "read":
            if self.cfg.cache_reads:
                st.read_cache[key] = result
            if not (isinstance(result, dict) and result.get("ok") is False):
                st.read_apps.add(app_of(action["name"]))
        if action["type"] == "write" and isinstance(result, dict) and result.get("ok"):
            if self.cfg.block_failed_repeats and result.get("done") is False:
                st.failed.add(key)           # ran, changed nothing: not "done before"
            else:
                st.writes_done.add(key)

    def _clock_for(self, param: str) -> str | None:
        if not self.clock:
            return None
        # "từ 9 giờ đến 10 giờ": a start takes the first time said, anything else the last (corrections come last)
        p = param.lower()
        return self.clock[0] if any(w in p for w in ("start", "from", "bat_dau")) else self.clock[-1]

    def _fix_name(self, name: str) -> str | None:
        if name in self.allowed_set:
            return name
        short = name.split(".")[-1]
        cands = [n for n in self.allowed if n.split(".", 1)[-1] == short]
        if len(cands) == 1:
            return cands[0]
        fname = _fold(name).replace(" ", "_")
        cands = [n for n in self.allowed if _fold(n) == fname]
        return cands[0] if len(cands) == 1 else None

    @staticmethod
    def _prefer_ref_id(spec, args: dict, st: GuardState) -> list[str]:
        """x-one-of [ID] | [raw value]: a raw value that a record read earlier holds becomes that record's ID, in the
        same place among the arguments. A value the user said that no record holds stays as it is."""
        refs = spec.ref_keys()
        ids = [g[0] for g in spec.one_of if len(g) == 1 and g[0] in refs]
        raws = [g[0] for g in spec.one_of if len(g) == 1 and g[0] not in refs]
        if any(k in args for k in ids):
            return []
        for idk in ids:
            known = st.rec_ids.get(app_of(refs[idk])) or {}
            for rk in raws:
                v = args.get(rk)
                rid = known.get(str(v)) if isinstance(v, (str, int)) and not isinstance(v, bool) else None
                if rid:
                    swapped = {(idk if k == rk else k): (rid if k == rk else x) for k, x in args.items()}
                    args.clear()
                    args.update(swapped)
                    return [f"{rk}->{idk}"]
        return []

    def check(self, action: dict, env, st: GuardState) -> Decision:
        name = self._fix_name(str(action.get("name") or ""))
        if name is None:
            return Decision("reject", action, f"tool '{action.get('name')}' không có trong danh sách được phép")
        spec = self.cat.tools[name]
        typ = KIND_TO_TYPE.get(spec.kind, "write")
        args = dict(action.get("arguments") or {})
        args = {k: nfc(v) for k, v in args.items() if v is not None}
        if self.cfg.drop_unknown_args and spec.params:
            args = {k: v for k, v in args.items() if k in spec.params}
        fb = self.formats
        if self.cfg.drop_unused_params and fb is not None and fb.tool_calls.get(name, 0) >= 20:
            keep = needed_params(spec)
            args = {k: v for k, v in args.items() if k in keep or fb.param_seen(name, k) > 0}
        if (self.cfg.drop_optional_text_unseen and typ == "write" and fb is not None
                and fb.tool_calls.get(name, 0) == 0):
            keep_title = any(w in spec.short for w in CREATES)
            args = {k: v for k, v in args.items()
                    if not (k in spec.params and spec.params[k].fuzzy and k not in spec.required
                            and not (keep_title and k in ("title", "name")))}
        if self.cfg.coerce_types:
            args = {k: coerce_value(v, spec.params[k]) if k in spec.params else v for k, v in args.items()}
        need_time = None
        if self.cfg.fix_time_formats and self.formats is not None:
            for k, v in list(args.items()):
                if k in spec.params and isinstance(v, str):
                    sh = self.formats.expected(name, k, spec.params[k].raw)
                    if not sh:
                        continue
                    v = coerce_shape(v, sh, self.ctx)
                    if sh in DATETIME_SHAPES and shape_of(v) == "date":
                        hhmm = self._clock_for(k)
                        if hhmm:                     # the model kept only the day
                            v = coerce_shape(f"{v}T{hhmm}", sh, self.ctx)
                        elif need_time is None:
                            need_time = (k, coerce_shape(f"{v}T09:30", sh, self.ctx))
                    args[k] = v
        if self.cfg.fix_names and self.formats is not None:
            for k, v in list(args.items()):
                if isinstance(v, str):
                    args[k] = canonical_name(v, self.formats.names(name, k), self.text)
        fixes = self._prefer_ref_id(spec, args, st) if self.cfg.prefer_ref_id and typ == "write" and spec.one_of else []
        fixed = {"type": typ, "name": name, "arguments": args}
        if need_time:
            k, example = need_time
            ask_tool = self.ask.tool_name if self.ask is not None else "sys.ask_missing_info"
            return Decision("reject", fixed, f"{k} cần cả ngày lẫn giờ, dạng {example}. Lấy giờ từ câu nói hoặc "
                                             f"dữ liệu đã đọc; nếu người dùng chưa nói giờ thì gọi {ask_tool} để hỏi")

        if typ == "ask" and self.ask is not None and self.ask.tool_name == name:
            tool, missing = self.ask.parse(args)
            if self.ask.tool_param:
                t2 = self._fix_name(str(tool or ""))
                if t2 is None or t2 in self.cat.meta_tools:
                    return Decision("reject", fixed, f"tool đích '{tool}' của câu hỏi không hợp lệ")
                tspec = self.cat.tools[t2]
                if self.cfg.fix_ask_params:
                    for i, m in enumerate(missing):
                        alt = None if m in tspec.params else ask_param(tspec, m)
                        if alt is None:
                            continue
                        # a field of a list item is usually something the model can fill in itself: sent back
                        # once with where to look, the same ask again goes out under the list's name
                        if alt[1] == "inside" and f"ask:{t2}.{m}" not in st.warned:
                            st.warned.add(f"ask:{t2}.{m}")
                            return Decision("reject", fixed, (
                                f"{m} nằm trong {alt[0]} của {t2} và do bạn điền: lấy từ lời người dùng, hoặc tra "
                                f"bằng tool đọc của ứng dụng đã cài (người dùng thường nhắc tới nơi có thông tin) "
                                f"rồi gọi {t2}. Tra không ra mới hỏi, và hỏi {alt[0]}"))
                        missing[i] = alt[0]
                bad = [m for m in missing if m not in tspec.params]
                if bad:
                    if not tspec.params:
                        return Decision("reject", fixed, f"{t2} không cần tham số nào, gọi thẳng {t2}")
                    if self.cfg.fix_ask_params:
                        return Decision("reject", fixed, (
                            f"{t2} không có tham số {bad}; tham số của {t2}: {', '.join(tspec.params)}. Chỉ hỏi về "
                            "tham số có trong danh sách này; thông tin khác người dùng nhắc tới thì tra bằng tool "
                            f"đọc của ứng dụng đã cài rồi gọi {t2}"))
                    return Decision("reject", fixed, f"{t2} không có tham số {bad}")
                missing = list(dict.fromkeys(missing))
                if (self.cfg.no_text_ask and missing and "text" not in st.warned
                        and all(m in TEXT_PARAMS and tspec.params[m].fuzzy for m in missing)):
                    st.warned.add("text")
                    return Decision("reject", fixed, f"{', '.join(missing)} là nội dung bạn tự viết từ lời người "
                                                     f"dùng: nếu người dùng đã nói ý cần ghi/nhắn thì gọi {t2} luôn; "
                                                     "chỉ hỏi lại khi họ chưa nói gì về nội dung")
                props = list(tspec.params)
                if self.cfg.ask_param_order == "schema" and props:
                    missing.sort(key=lambda m: props.index(m) if m in props else len(props))
                elif self.cfg.ask_param_order == "required":
                    req = tspec.required
                    missing.sort(key=lambda m: (req.index(m) if m in req else len(req),
                                                props.index(m) if m in props else len(props)))
                fixed["arguments"] = self.ask.build(t2, missing) | {
                    k: v for k, v in args.items() if k not in (self.ask.tool_param, self.ask.missing_param)}
                if self.cfg.no_ask_single_hit and "hit" not in st.warned:
                    for m in missing:
                        ref = tspec.ref_keys().get(m)
                        if ref and st.hits.get(ref) == 1:
                            st.warned.add("hit")
                            return Decision("reject", fixed, f"lần đọc gần nhất chỉ có một kết quả cho {m} ({ref}): "
                                                             f"nếu đó đúng là thứ người dùng nói thì dùng ID đó và "
                                                             f"gọi {t2}, không hỏi lại")
                if self.cfg.ask_ids_after_read:
                    refs = tspec.ref_keys()
                    for m in missing:
                        rapp = app_of(refs[m]) if m in refs else None
                        rd = self.readers(rapp) if rapp else []
                        if rd and rapp not in st.read_apps:
                            return Decision("reject", fixed, f"{m} là ID ({refs[m]}) mà chưa tra {rapp}: gọi "
                                                             f"{' hoặc '.join(rd[:2])} để tìm trước, tra không ra mới hỏi",
                                            call=self.finder(rd))
                need = needed_params(tspec) if self.cfg.ask_needed_only else set()
                asked = self.formats.asked if self.formats is not None else set()
                if (need and missing and not any(m in need for m in missing)
                        and not all(f"{t2}.{m}" in asked for m in missing)):
                    return Decision("reject", fixed, f"{', '.join(missing)} không bắt buộc với {t2}; đã đủ thông tin "
                                                     f"bắt buộc thì gọi {t2} luôn", call=t2)

        # IDs must come from data we actually read (tool results or history)
        if self.cfg.check_xref:
            for k, ref in spec.ref_keys().items():
                if k not in fixed["arguments"]:
                    continue
                vals = fixed["arguments"][k]
                vals = vals if isinstance(vals, list) else [vals]
                miss = [v for v in vals if str(v) not in st.seen]
                rd = self.readers(app_of(ref))
                tools = f" ({' hoặc '.join(rd[:2])})" if rd else ""
                if miss:
                    return Decision("reject", fixed,
                                    f"{k}={miss[0]!r} chưa xuất hiện trong kết quả nào; hãy đọc dữ liệu ({ref}) "
                                    f"bằng tool phù hợp{tools} để lấy đúng ID trước", call=self.finder(rd))
                if self.cfg.xref_typed and rd:
                    own = st.seen_by_app.get(app_of(ref), set()) | st.seen_by_key.get(k, set())
                    wrong = [v for v in vals if str(v) not in own]
                    if wrong:
                        return Decision("reject", fixed, f"{k}={wrong[0]!r} không phải ID của {app_of(ref)} ({ref}); "
                                                         f"lấy đúng ID từ kết quả của {' hoặc '.join(rd[:2])}",
                                        call=self.finder(rd))

        key = canon_key(name, fixed["arguments"])
        if typ == "read" and self.cfg.cache_reads and key in st.read_cache:
            return Decision("cached", fixed, "repeated read", st.read_cache[key])
        if typ == "read" and key in st.history_cache:
            return Decision("cached", fixed, "read in history", st.history_cache[key])
        if typ == "write" and key in st.failed:
            return Decision("reject", fixed, "lệnh này vừa chạy mà không có tác dụng với đúng các tham số này; lấy "
                                             "lại giá trị/ID đúng từ dữ liệu rồi mới gọi lại")
        repeatable = self.formats is not None and name in self.formats.repeatable
        if typ == "write" and self.cfg.block_duplicate_writes and key in st.writes_done and not repeatable:
            return Decision("cached", fixed, "repeated write", {"ok": True, "note": "lệnh này đã thực hiện trước đó"})

        try:
            # validate() is not logged, so a bad call never reaches the submission
            v = env.validate(name, fixed["arguments"])
        except Exception as ex:
            v = {"ok": False, "error": str(ex)}
        if isinstance(v, dict) and not v.get("ok", False):
            return Decision("reject", fixed, str(v.get("error") or "sai schema"))
        return Decision("execute", fixed, fixes=fixes)
