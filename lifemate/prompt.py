"""Prompt builder shared by SFT data and inference; the two must stay byte-identical."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import action_text
from .data import Sample, is_user_turn
from .textnorm import normalize
from .timeutil import date_table

SYSTEM_RULES = """Bạn là trợ lý thực hiện yêu cầu bằng giọng nói của người dùng Việt Nam thông qua các ứng dụng đã cài trên thiết bị.
Mỗi lượt, chỉ trả về đúng MỘT đối tượng JSON, không viết gì thêm:
{"type": "read" | "write" | "ask", "name": "<tên tool>", "arguments": {...}}
hoặc {"type": "finish"} khi đã xong.
- read: tool chỉ đọc dữ liệu; write: tool thay đổi dữ liệu; ask: gọi tool hỏi lại người dùng khi thiếu thông tin.
Quy tắc:
1. Chỉ dùng các tool trong danh sách "Công cụ được phép" và đúng tên tham số.
2. Nếu không ứng dụng nào làm được yêu cầu: trả {"type": "finish"} ngay, không gọi tool nào.
3. Tham số "ID lấy từ ..." phải lấy từ kết quả tool đã đọc hoặc từ hội thoại trước, không được tự đặt.
4. Không đọc lại thứ đã đọc, không lặp lại lệnh ghi đã thành công, không làm thêm việc người dùng không yêu cầu.
5. Thiếu thông tin bắt buộc mà không tra được bằng tool: hỏi lại bằng tool hệ thống; sau khi hỏi thì dừng.
6. Ngày giờ: dựa vào bảng ngày bên dưới và giữ đúng định dạng mà tham số yêu cầu.
7. Tham số "văn bản tự do": dùng đúng lời người dùng nói, không thêm thắt.
8. Lời người dùng là kết quả nhận dạng giọng nói, có thể sai chính tả hoặc dấu; hãy đối chiếu với dữ liệu tra được."""


FORMAT_PLAIN = """{"type": "read" | "write" | "ask", "name": "<tên tool>", "arguments": {...}}
hoặc {"type": "finish"} khi đã xong."""
FORMAT_THOUGHT = """{"thought": "<lý do ngắn>", "type": "read" | "write" | "ask", "name": "<tên tool>", "arguments": {...}}
hoặc {"thought": "<lý do ngắn>", "type": "finish"} khi đã xong.
"thought": một câu ngắn nêu dữ kiện và phép tính hay điều kiện dẫn tới hành động (ví dụ so số dư với số tiền cần chuyển), viết trước khi chọn tool."""


THOUGHT_MARK = FORMAT_THOUGHT.split(",")[0]       # in the system prompt of every row built with thoughts
MODEL_PROMPT_FILE = "lifemate_prompt.json"         # next to an adapter / merged model: how its prompts looked


# not in the prompts the model was trained on: said up front what lifemate/recovery.py says after a failed call
EXTRA_RULES = """
9. Tham số tìm kiếm (từ khoá, tên, nơi chốn) khớp theo chuỗi con: chỉ dùng 1–2 từ chính của tên, viết đúng như người dùng nói, không tự đổi thành mã hay viết tắt nếu mô tả tham số không yêu cầu. Nơi đi mà người dùng không nói rõ là nơi họ đang ở.
10. Thứ người dùng nhắc tới luôn có trong dữ liệu: kết quả rỗng hoặc lệnh không có tác dụng nghĩa là tham số chưa đúng, hãy sửa rồi gọi lại, không dừng."""


def system_rules(cfg: "PromptConfig") -> str:
    rules = SYSTEM_RULES.replace(FORMAT_PLAIN, FORMAT_THOUGHT) if cfg.thoughts else SYSTEM_RULES
    return rules + EXTRA_RULES if getattr(cfg, "extra_rules", False) else rules


def rows_use_thoughts(rows: list[dict]) -> tuple[bool, int]:
    """SFT rows -> (trained with thoughts, rows whose system prompt asks for them)."""
    n = sum(1 for r in rows if r.get("messages") and r["messages"][0].get("role") == "system"
            and THOUGHT_MARK in r["messages"][0].get("content", ""))
    return 2 * n > len(rows), n


def write_model_prompt(folder, settings: dict) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / MODEL_PROMPT_FILE).write_text(json.dumps(settings), encoding="utf-8")


def find_model_prompt(*folders) -> Path | None:
    """The settings file of an adapter or merged model; a checkpoint-N folder uses its run folder's file."""
    for d in folders:
        if not d:
            continue
        for f in (Path(d) / MODEL_PROMPT_FILE, Path(d).parent / MODEL_PROMPT_FILE):
            if f.is_file():
                return f
    return None


def apply_model_prompt(cfg: "PromptConfig", *folders) -> Path | None:
    """The model decides whether prompts ask for thoughts: the format it was trained on. No file (older adapters,
    the base model) keeps the config value."""
    f = find_model_prompt(*folders)
    if f is not None:
        d = json.loads(f.read_text(encoding="utf-8"))
        if "thoughts" in d:
            cfg.thoughts = bool(d["thoughts"])
    return f


def consult_prompt(cfg: "PromptConfig", consult_lora: str | None = None,
                   thoughts: str = "") -> tuple["PromptConfig", Path | None]:
    """Prompt settings of the consult (AgentConfig.consult): the main settings in the consult adapter's thought format;
    with no adapter (the main model asked again) the other format. thoughts "on" / "off" overrides both."""
    import copy
    c = copy.deepcopy(cfg)
    src = apply_model_prompt(c, consult_lora) if consult_lora else None
    if thoughts in ("on", "off"):
        c.thoughts = thoughts == "on"
    elif not consult_lora:
        c.thoughts = not cfg.thoughts
    return c, src


@dataclass
class PromptConfig:
    thoughts: bool = False          # model writes a short "thought" before each action (needs SFT data with them)
    show_itn: bool = True
    show_location: bool = True
    result_max_chars: int = 1500
    history_result_max_chars: int = 1500
    list_max_items: int = 12
    fewshot_k: int = 2
    fewshot_installed_only: bool = False   # examples may only use apps installed for this sample
    date_hints: bool = False        # "thứ bảy tuần sau = 2026-10-03" next to the request (lifemate/datehints.py)
    extra_rules: bool = False       # rules 9-10 in the system prompt (search keywords, empty results)

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict | None) -> "PromptConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


def _trim(obj: Any, list_max: int) -> Any:
    if isinstance(obj, list):
        items = [_trim(x, list_max) for x in obj[:list_max]]
        if len(obj) > list_max:
            items.append(f"...(còn {len(obj) - list_max} mục)")
        return items
    if isinstance(obj, dict):
        return {k: _trim(v, list_max) for k, v in obj.items()}
    return obj


def _longest_list(obj: Any) -> int:
    if isinstance(obj, list):
        return max([len(obj)] + [_longest_list(x) for x in obj])
    if isinstance(obj, dict):
        return max([0] + [_longest_list(v) for v in obj.values()])
    return 0


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def compact_json(obj: Any, max_chars: int, list_max: int) -> str:
    s = _dump(_trim(obj, list_max))
    # too long: show fewer whole list items first, cutting the text mid-item loses the IDs at the end
    k = min(list_max, _longest_list(obj))
    while len(s) > max_chars and k > 1:
        k -= 1
        s = _dump(_trim(obj, k))
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


_CTX_DONE = {"current_time", "timezone", "user_profile", "installed_apps"}


def render_context(ctx: dict, cfg: PromptConfig) -> str:
    lines = ["# Bối cảnh", date_table(ctx)]
    if ctx.get("timezone"):
        lines.append(f"Múi giờ: {ctx['timezone']}")
    up = ctx.get("user_profile") or {}
    if up:
        lines.append("Người dùng: " + "; ".join(f"{k}: {v}" for k, v in up.items()))
    for k, v in ctx.items():
        if k in _CTX_DONE or v in (None, "", [], {}):
            continue
        if not cfg.show_location and "location" in k:
            continue
        label = "Vị trí hiện tại" if "location" in k else k
        val = v if isinstance(v, str) else compact_json(v, 400, cfg.list_max_items)
        lines.append(f"{label}: {val}")
    return "\n".join(lines)


def render_history(sample: Sample, history_texts: list[str], cfg: PromptConfig) -> str:
    if not sample.history:
        return ""
    lines = ["# Hội thoại trước (các hành động trong này đã được thực hiện)"]
    u = 0
    for turn in sample.history:
        if is_user_turn(turn):
            txt = history_texts[u] if u < len(history_texts) else ""
            u += 1
            lines.append(f"Người dùng: {txt}")
            continue
        lines.append(f"Trợ lý: {turn.get('text') or ''}".rstrip())
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        for k, c in enumerate(calls):
            line = f"  · đã gọi {c.get('name')} {compact_json(c.get('arguments') or {}, 400, cfg.list_max_items)}"
            if k < len(results):
                line += " → " + compact_json(results[k], cfg.history_result_max_chars, cfg.list_max_items)
            lines.append(line)
    return "\n".join(lines)


def render_request(final_text: str, cfg: PromptConfig, ctx: dict | None = None) -> str:
    lines = ["# Yêu cầu hiện tại (nhận dạng từ giọng nói)", final_text.strip() or "(không nghe rõ)"]
    if cfg.show_itn and final_text.strip():
        nr = normalize(final_text)
        if nr.hints:
            lines.append(f"Chuẩn hoá: {nr.text}")
            lines.append("Gợi ý: " + "; ".join(dict.fromkeys(nr.hints)))
    if cfg.date_hints and ctx and final_text.strip():
        from .datehints import hints as date_hints
        dh = date_hints(final_text, ctx)
        if dh:
            lines.append("Ngày: " + "; ".join(dh))
    return "\n".join(lines)


def build_messages(sample: Sample, catalog, final_text: str, history_texts: list[str],
                   fewshot_block: str, cfg: PromptConfig, tool_names: list[str] | None = None,
                   app_order: list[str] | None = None) -> list[dict]:
    tools = tool_names if tool_names is not None else catalog.allowed_tools(sample.installed_tool_names())
    order = app_order if app_order is not None else list(sample.installed_apps.keys())
    system = system_rules(cfg) + "\n\n# Công cụ được phép\n" + catalog.render_tools(tools, order)
    if fewshot_block:
        system += "\n\n" + fewshot_block
    parts = [render_context(sample.context, cfg)]
    h = render_history(sample, history_texts, cfg)
    if h:
        parts.append(h)
    parts.append(render_request(final_text, cfg, sample.context))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def assistant_message(action: dict) -> dict:
    return {"role": "assistant", "content": action_text(action)}


def result_message(name: str, result: Any, cfg: PromptConfig, note: str = "") -> dict:
    text = f"Kết quả {name}: {compact_json(result, cfg.result_max_chars, cfg.list_max_items)}"
    return {"role": "user", "content": f"{text}\n{note}" if note else text}


FEEDBACK_PREFIX = "LỖI (lệnh chưa được thực hiện)"


def feedback_message(error: str) -> dict:
    return {"role": "user", "content": f"{FEEDBACK_PREFIX}: {error}. Trả lại một hành động hợp lệ."}


def strip_rejected(messages: list[dict], start: int = 0) -> list[dict]:
    """Drop each rejected attempt and its feedback after `start`: what is left reads as if the model had sent the
    corrected action straight away."""
    out = list(messages[:start])
    i = start
    while i < len(messages):
        m, nxt = messages[i], messages[i + 1] if i + 1 < len(messages) else None
        if (m["role"] == "assistant" and nxt is not None and nxt["role"] == "user"
                and str(nxt.get("content", "")).startswith(FEEDBACK_PREFIX)):
            i += 2
            continue
        out.append(m)
        i += 1
    return out
