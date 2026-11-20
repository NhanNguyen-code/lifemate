"""What the model is told when a call ran but found or changed nothing.

In the train labels a read finds something (4 of 1599 results are empty) and a write takes effect (1 of 1725 does
not), and an ID is only asked for after a search with two or more hits. The data a request needs is there, so an empty
search or a write with done=false nearly always means wrong arguments: a long keyword, a code the user never said, a
place or a day taken from habit. The model gets a note that says what to change, and one more try before it may stop.

On public data the model stopped again after that note in 21 of 26 cases, so the retry is no longer left to it
(AgentConfig.recover_auto / recover_force): at a stop right after an empty search the agent searches again itself where
it can (relaxed), and otherwise writes the start of the same call, with what to change in the thought, for the model
to finish (retry_thought, retry_prefix). A read cannot change the state, so the cost is at most a few extra reads.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .guard import TITLES
from .textnorm import strip_diacritics
from .timeutil import shape_of

_META = {"ok", "done", "message", "note", "error"}
# where a trip starts ("origin", "departure_city"); not from_currency, from_account_id, date_from
_PLACE = re.compile(r"^(origin|from|pickup|departure|depart|diem_di|noi_di)(_(city|place|point|location|station|"
                    r"stop|address|name))?$|^start_(place|point|location|station|stop|city|address)$", re.I)


def outcome(action_type: str, args: dict | None, result: Any) -> str | None:
    """"noop": the tool ran and did nothing (done=false). "empty": a read with filters that found nothing.
    None: anything else, also a read without arguments that lists nothing (that is the data, not a mistake)."""
    if not isinstance(result, dict) or result.get("ok") is False:
        return None
    if result.get("done") is False:
        return "noop"
    if action_type != "read" or not args:
        return None
    payload = [v for k, v in result.items() if k not in _META]
    if payload and all(isinstance(v, list) and not v for v in payload):
        return "empty"
    return None


def _text_keys(spec, args: dict) -> list[str]:
    out = []
    for k, v in args.items():
        p = spec.params.get(k) if spec is not None else None
        if not isinstance(v, str) or not v.strip() or (p is not None and (p.enum or p.ref or p.fmt)):
            continue
        if shape_of(v) != "text":
            continue
        out.append(k)
    return out


def _date_keys(args: dict) -> list[str]:
    return [k for k, v in args.items() if isinstance(v, str) and shape_of(v) != "text"]


def _origin_keys(spec, keys: list[str]) -> list[str]:
    """Start-of-trip parameters that take a city or a station. A free-text one takes an address ("nhà" ->
    home_address), which the model fills in from the profile already."""
    return [k for k in keys if _PLACE.search(k)
            and not (spec is not None and k in spec.params and spec.params[k].fuzzy)]


def hint(kind: str, spec, args: dict, ctx: dict | None, readers: list, message: str = "") -> str:
    """Vietnamese note appended to the tool result. spec: ToolSpec of the call; readers: ToolSpecs of the read tools
    of the same app."""
    name = spec.name if spec is not None else "tool"
    others = [r for r in readers if r.name != name]
    if kind == "noop":
        why = f" ({message})" if message else ""
        look = sorted(others or readers, key=lambda r: len(r.required))[:3]      # list-all tools first
        how = (f"Lấy giá trị/ID đúng bằng {' hoặc '.join(r.name for r in look)}" if look
               else "Xem lại dữ liệu đã đọc để lấy giá trị/ID đúng")
        return (f"Lưu ý: lệnh chưa có tác dụng{why}. Dữ liệu người dùng nhắc tới thường có sẵn, nên nhiều khả năng "
                f"tham số chưa đúng. {how} rồi gọi lại {name}.")
    tips = []
    optional = [k for k in args if spec is None or k not in spec.required]
    if optional and len(optional) == len(args):
        tips.append(f"gọi lại không kèm {', '.join(optional)} để xem cả danh sách rồi chọn mục gần đúng nhất")
    elif optional:
        tips.append(f"bỏ bớt điều kiện không bắt buộc ({', '.join(optional)})")
    text = _text_keys(spec, args)
    if text:
        coded = spec is not None and any("mã" in spec.params[k].description.lower().split() for k in text
                                         if k in spec.params)
        tips.append(f"{', '.join(text)}: chỉ dùng 1–2 từ chính của tên, viết đúng như người dùng nói"
                    + ("" if coded else ", không tự đổi thành mã hay viết tắt"))
        up = (ctx or {}).get("user_profile") or {}
        city = up.get("city") or up.get("current_city")
        if city and _origin_keys(spec, text):
            tips.append(f"nơi đi mà người dùng không nói rõ là nơi họ đang ở ({city})")
    dates = _date_keys(args)
    if dates:
        tips.append(f"{', '.join(dates)}: đối chiếu lại với bảng ngày ở phần Bối cảnh")
    same = [r.name for r in others if spec is not None and r.returns and r.returns == spec.returns and not r.required]
    if same:
        tips.append(f"hoặc xem danh sách bằng {' hoặc '.join(same[:2])}")
    body = "; ".join(tips) if tips else "xem lại từng tham số"
    return ("Lưu ý: không có kết quả. Dữ liệu người dùng nhắc tới thường có sẵn, nên nhiều khả năng tham số chưa "
            f"khớp. Cách sửa: {body}. Hãy gọi lại với tham số đã sửa; chỉ dừng khi người dùng chỉ hỏi thông tin và "
            "kết quả rỗng chính là câu trả lời.")


def _fold(s: str) -> str:
    return " ".join(strip_diacritics(str(s)).lower().split())


_ADDRESS = TITLES | {"mr", "mrs", "ms", "miss", "dr", "sir"}      # "Mr. Johnson" is stored as "Johnson" as well


def _no_title(v: str) -> str:
    """Without the words of address in front of a name: "anh Tuấn" -> "Tuấn", "cô giáo Mai" -> "Mai". Unchanged
    when no such word leads or nothing would be left."""
    words = v.split()
    k = 0
    while k < min(2, len(words) - 1) and _fold(words[k]).rstrip(".") in _ADDRESS:
        k += 1
    return " ".join(words[k:]) if k else v


def relaxed(spec, args: dict) -> list[tuple[dict, str]]:
    """Calls the agent may make itself after a read found nothing, most specific first:
    "title": the same search without the word of address in front of a name (data stores "Tuấn", not "anh Tuấn");
    "all":   the list without filters, when every filter was optional.
    A search whose required arguments are wrong has no such call: the model has to write it again."""
    out: list[tuple[dict, str]] = []
    if spec is None or not args:
        return out
    cut = {k: _no_title(args[k]) for k in _text_keys(spec, args)}
    cut = {k: v for k, v in cut.items() if v != args[k]}
    if cut:
        out.append(({**args, **cut}, "title"))
    need = set(spec.required) | {p for g in spec.one_of for p in g}
    if not any(k in need for k in args):
        out.append(({}, "all"))
    return out


def auto_thought(how: str, name: str) -> str:
    if how == "title":
        return f"Tìm không ra, có thể do từ xưng hô đứng trước tên; gọi lại {name} chỉ với tên."
    return f"Tìm theo điều kiện đó không ra; gọi {name} không kèm điều kiện để xem cả danh sách."


def auto_note(how: str, found: bool) -> str:
    if not found:
        return "Lưu ý: vẫn không có kết quả."
    if how == "all":
        return ("Lưu ý: đây là cả danh sách, vì tìm theo điều kiện trước đó không ra. Tên trong lời người dùng có "
                "thể bị nghe nhầm hoặc gọi khác đi: chọn mục gần đúng nhất rồi làm tiếp yêu cầu; chỉ hỏi lại khi "
                "không mục nào gần đúng.")
    return "Lưu ý: lần tìm lại này có kết quả, làm tiếp yêu cầu với dữ liệu trên."


def retry_thought(spec, args: dict, ctx: dict | None, said: str = "") -> str:
    """The thought the forced retry starts from: what to change, in the model's own voice. `said`: what the user
    said, to tell a place they named from one the model filled in from habit."""
    name = spec.name if spec is not None else "tool"
    tips = []
    text = _text_keys(spec, args)
    up = (ctx or {}).get("user_profile") or {}
    city = up.get("city") or up.get("current_city")
    origin = _origin_keys(spec, text)
    heard = f" {_fold(said)} "                 # whole words: "han" is not in "tháng"
    if origin and city and not any(f" {_fold(args[k])} " in heard for k in origin):
        tips.append(f"người dùng không nói nơi đi nên nơi đi là nơi họ đang ở ({city}), nơi đến là nơi họ muốn tới")
    if text:
        coded = spec is not None and any("mã" in spec.params[k].description.lower().split() for k in text
                                         if k in spec.params)
        tips.append("tên và từ khoá viết đúng như người dùng nói, chỉ 1–2 từ chính"
                    + ("" if coded else ", không đổi thành mã hay viết tắt"))
    if _date_keys(args) and not (origin and tips and "nơi đi" in tips[0]):
        tips.append("ngày đối chiếu lại với bảng ngày")      # not next to a wrong place: one thing to change at a time
    how = "; ".join(tips) if tips else "xem lại từng tham số"
    return f"Không có kết quả nên tham số chưa khớp: {how}. Sửa rồi gọi lại {name}."


def named_thought(name: str, typ: str) -> str:
    """The thought a call the guard named starts from (AgentConfig.force_lookup)."""
    if typ == "read":
        return f"Chưa có ID cần dùng nên phải tra bằng {name} trước, rồi mới làm tiếp yêu cầu."
    return f"Đã đủ thông tin bắt buộc nên gọi {name} luôn, không hỏi thêm."


def retry_prefix(thought: str | None, typ: str, name: str) -> str:
    """Start of the action the model has to finish: same bytes as actions.action_text would write."""
    head = f'{{"thought": {json.dumps(thought, ensure_ascii=False)}, ' if thought else "{"
    return head + f'"type": "{typ}", "name": "{name}", "arguments": {{'


def nudge(fail: dict) -> str:
    """Feedback for a finish / ask sent right after a failed call that was never tried again."""
    what = "chưa có tác dụng" if fail.get("kind") == "noop" else "không tìm thấy gì"
    return (f"{fail.get('name')} vừa {what} và chưa được thử lại. Dữ liệu cần dùng thường có sẵn: hãy thử lại một lần "
            "với tham số đã sửa theo lưu ý ở trên (hoặc dùng tool đọc khác của ứng dụng đó). Chỉ khi thử lại vẫn "
            "không được mới dừng")


# a finish whose thought says the user left something out. In the labels missing information is looked up (another
# app holds it) or asked for with the system tool; a sample never just ends there
LOST = re.compile(r"(chưa|không) (cung cấp|chỉ định|nói rõ|nêu rõ|đề cập|ghi rõ)|thiếu thông tin|"
                  r"(không|chưa) (có|đủ) thông tin", re.I)
LOST_FEEDBACK = ("Thiếu thông tin thì chưa dừng: thứ người dùng nhắc tới thường nằm ở một ứng dụng khác đã cài (tin "
                 "nhắn, email, ghi chú, lịch, danh bạ, giao dịch...), hãy tra tiếp bằng tool đọc phù hợp rồi làm yêu "
                 "cầu. Tra hết mà vẫn thiếu thông tin bắt buộc thì hỏi lại bằng tool hệ thống")


def lost(thought: str | None) -> bool:
    return bool(thought) and bool(LOST.search(thought))


REPEAT_NOTE = ("Lưu ý: lệnh này đã gọi với đúng các tham số này, kết quả không đổi. Đổi tham số hoặc làm bước tiếp "
               "theo.")


def ask_deferred_note(tool: str | None) -> str:
    t = f" Không gọi {tool} khi chưa có câu trả lời." if tool else ""
    return ("Lưu ý: câu hỏi sẽ được gửi cho người dùng sau cùng. Nếu trong yêu cầu còn việc khác làm được ngay mà "
            f"không cần câu trả lời này thì làm tiếp; không còn thì trả finish.{t} Không hỏi thêm câu khác.")
