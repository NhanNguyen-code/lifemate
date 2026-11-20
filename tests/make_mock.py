"""Tiny fake dataset (train + public_test) with the official layout, for CPU tests."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NOW = "2026-10-20T19:30:00+07:00"


def S(desc, **kw):
    d = {"type": "string", "description": desc}
    d.update(kw)
    return d


APPS = {
    "contacts": {
        "name_vi": "Danh bạ", "group": "communication", "description": "Tra cứu danh bạ.",
        "tools": [
            {"name": "contacts.search", "kind": "read", "description": "Tìm liên hệ theo tên.",
             "parameters": {"type": "object", "properties": {
                 "query": S("Tên hoặc một phần tên liên hệ")}, "required": []},
             "returns": "results: danh sách {id, name, phone}"},
            {"name": "contacts.get", "kind": "read", "description": "Xem chi tiết một liên hệ.",
             "parameters": {"type": "object", "properties": {
                 "contact_id": S("ID liên hệ", **{"x-ref": "contacts.contacts"})},
                 "required": ["contact_id"]},
             "returns": "contact"},
        ]},
    "sms": {
        "name_vi": "Tin nhắn", "group": "communication", "description": "Gửi và đọc tin nhắn SMS.",
        "tools": [
            {"name": "sms.send", "kind": "write", "description": "Gửi tin nhắn SMS.",
             "parameters": {"type": "object", "properties": {
                 "contact_id": S("ID người nhận", **{"x-ref": "contacts.contacts"}),
                 "phone_number": S("Số điện thoại người nhận"),
                 "content": S("Nội dung tin nhắn", **{"x-match": "fuzzy"})},
                 "required": ["content"], "x-one-of": [["contact_id"], ["phone_number"]]},
             "returns": "message_id"},
            {"name": "sms.list", "kind": "read", "description": "Xem tin nhắn gần đây.",
             "parameters": {"type": "object", "properties": {
                 "limit": {"type": "integer", "minimum": 1, "description": "Số tin"}},
                 "required": []},
             "returns": "messages"},
        ]},
    "smart_home": {
        "name_vi": "Nhà thông minh", "group": "home", "description": "Điều khiển thiết bị trong nhà.",
        "tools": [
            {"name": "smart_home.get_status", "kind": "read", "description": "Xem trạng thái thiết bị.",
             "parameters": {"type": "object", "properties": {}, "required": []},
             "returns": "devices: danh sách {id, name, room, type, power, brightness}"},
            {"name": "smart_home.light_set", "kind": "write", "description": "Bật/tắt, chỉnh độ sáng đèn.",
             "parameters": {"type": "object", "properties": {
                 "device_id": S("ID đèn", **{"x-ref": "smart_home.devices"}),
                 "power": {"type": "string", "enum": ["on", "off"], "description": "Bật hay tắt"},
                 "brightness": {"type": "integer", "minimum": 0, "maximum": 100,
                                "description": "Độ sáng %"}},
                 "required": ["device_id", "power"]},
             "returns": "device"},
        ]},
    "calendar": {
        "name_vi": "Lịch", "group": "productivity", "description": "Quản lý lịch hẹn.",
        "tools": [
            {"name": "calendar.create_event", "kind": "write", "description": "Tạo sự kiện.",
             "parameters": {"type": "object", "properties": {
                 "title": S("Tiêu đề", **{"x-match": "fuzzy"}),
                 "start_time": S("Thời điểm bắt đầu, ISO 8601, ví dụ 2026-10-21T08:00:00+07:00"),
                 "end_time": S("Thời điểm kết thúc, ISO 8601")},
                 "required": ["title", "start_time"]},
             "returns": "event_id"},
            {"name": "calendar.list_events", "kind": "read", "description": "Xem sự kiện theo ngày.",
             "parameters": {"type": "object", "properties": {
                 "date": S("Ngày YYYY-MM-DD")}, "required": ["date"]},
             "returns": "events"},
        ]},
    "banking": {
        "name_vi": "Ngân hàng", "group": "finance", "description": "Tài khoản ngân hàng.",
        "tools": [
            {"name": "banking.get_balance", "kind": "read", "description": "Xem số dư.",
             "parameters": {"type": "object", "properties": {}, "required": []},
             "returns": "balance"},
            {"name": "banking.transfer", "kind": "write", "description": "Chuyển tiền cho liên hệ.",
             "parameters": {"type": "object", "properties": {
                 "contact_id": S("ID người nhận", **{"x-ref": "contacts.contacts"}),
                 "amount": {"type": "integer", "minimum": 1000, "description": "Số tiền VND"},
                 "note": S("Lời nhắn", **{"x-match": "fuzzy"})},
                 "required": ["contact_id", "amount"]},
             "returns": "transfer_id, balance"},
        ]},
    "ride": {
        "name_vi": "Gọi xe", "group": "transport", "description": "Đặt xe công nghệ.",
        "tools": [
            {"name": "ride.estimate", "kind": "read", "description": "Ước tính giá cước.",
             "parameters": {"type": "object", "properties": {
                 "destination": S("Điểm đến")}, "required": ["destination"]},
             "returns": "price"},
            {"name": "ride.book", "kind": "write", "description": "Đặt xe.",
             "parameters": {"type": "object", "properties": {
                 "destination": S("Điểm đến", **{"x-match": "fuzzy"}),
                 "pickup": S("Điểm đón"),
                 "vehicle": {"type": "string", "enum": ["bike", "car"], "description": "Loại xe"}},
                 "required": ["destination"]},
             "returns": "booking_id"},
        ]},
}

SYS_APP = {
    "name_vi": "Hệ thống", "group": "system", "description": "Công cụ hệ thống.",
    "tools": [
        {"name": "sys.ask_missing_info", "kind": "meta",
         "description": "Hỏi lại người dùng khi thiếu thông tin bắt buộc.",
         "parameters": {"type": "object", "properties": {
             "tool_name": S("Tool định gọi"),
             "params": {"type": "array", "items": {"type": "string"},
                        "description": "Các tham số còn thiếu"}},
             "required": ["tool_name", "params"]},
         "returns": ""},
    ]}

BASE_STATE = {
    "contacts": {"contacts": [
        {"id": "c_lan", "name": "Chị Lan", "phone": "0901111222"},
        {"id": "c_minh", "name": "Anh Minh", "phone": "0903333444"},
        {"id": "c_me", "name": "Mẹ", "phone": "0905555666"},
        {"id": "c_hang", "name": "Hằng", "phone": "0907777888"}]},
    "sms": {"messages": []},
    "smart_home": {"devices": [
        {"id": "light_living", "name": "Đèn phòng khách", "room": "phòng khách", "type": "light",
         "power": "on", "brightness": 80},
        {"id": "light_bedroom", "name": "Đèn phòng ngủ", "room": "phòng ngủ", "type": "light",
         "power": "off", "brightness": 0}]},
    "calendar": {"events": []},
    "banking": {"balance": 5000000, "transfers": []},
    "ride": {"bookings": []},
}

CONTEXT = {
    "current_time": NOW, "timezone": "Asia/Ho_Chi_Minh",
    "current_location": {"lat": 10.7769, "lng": 106.7009, "address": "Phố đi bộ Nguyễn Huệ, Quận 1"},
    "user_profile": {"name": "Nguyễn Văn An", "home_address": "12 Nguyễn Huệ, Quận 1, TP.HCM",
                     "workplace": "Tòa nhà Bitexco, Quận 1", "current_city": "TP. Hồ Chí Minh"},
}

C = lambda name, **a: {"name": name, "arguments": a}

TRAIN = [
    ("train_0001", ["contacts", "sms", "smart_home"], "nhắn cho chị lan là tối nay em về muộn",
     [C("contacts.search", query="Lan"), C("sms.send", contact_id="c_lan", content="tối nay em về muộn")], []),
    ("train_0002", ["smart_home", "contacts"], "tắt đèn phòng khách giúp mình",
     [C("smart_home.get_status"), C("smart_home.light_set", device_id="light_living", power="off")], []),
    ("train_0003", ["calendar", "contacts"], "đặt lịch họp với team lúc tám giờ sáng mai",
     [C("calendar.create_event", title="họp với team", start_time="2026-10-21T08:00:00+07:00")], []),
    ("train_0004", ["contacts", "sms"], "nhắn tin cho anh minh",
     [C("contacts.search", query="Minh"),
      C("sys.ask_missing_info", tool_name="sms.send", params=["content"])], []),
    ("train_0005", ["contacts", "sms", "calendar"], "gọi xe grab về nhà cho mình", [], []),
    ("train_0006", ["contacts", "banking"], "hai trăm rưỡi nghìn nhé",
     [C("banking.transfer", contact_id="c_me", amount=250000)],
     [{"role": "user", "audio_text": "chuyển tiền cho mẹ"},
      {"role": "assistant", "text": "Bạn muốn chuyển bao nhiêu tiền cho mẹ?",
       "tool_calls": [C("contacts.search", query="mẹ"),
                      C("sys.ask_missing_info", tool_name="banking.transfer", params=["amount"])]}]),
    ("train_0007", ["smart_home"], "bật đèn phòng ngủ độ sáng năm mươi phần trăm",
     [C("smart_home.get_status"),
      C("smart_home.light_set", device_id="light_bedroom", power="on", brightness=50)], []),
    ("train_0008", ["banking", "contacts"], "kiểm tra số dư tài khoản giúp tôi",
     [C("banking.get_balance")], []),
    ("train_0009", ["contacts", "banking", "sms"], "chuyển cho hằng một triệu hai tiền ăn",
     [C("contacts.search", query="Hằng"),
      C("banking.transfer", contact_id="c_hang", amount=1200000, note="tiền ăn")], []),
    ("train_0010", ["calendar"], "tối thứ sáu này bảy giờ rưỡi nhắc mình đi ăn sinh nhật",
     [C("calendar.create_event", title="đi ăn sinh nhật", start_time="2026-10-23T19:30:00+07:00")], []),
    ("train_0011", ["smart_home", "banking"], "đặt bàn nhà hàng tối nay", [], []),
    ("train_0012", ["contacts", "sms"], "gửi tin cho số không chín không một một một một hai hai hai là anh tới rồi",
     [C("sms.send", phone_number="0901111222", content="anh tới rồi")], []),
]

PUBLIC = [
    ("public_0001", ["ride", "contacts"], "đặt xe máy về nhà hàng ngon"),
    ("public_0002", ["smart_home"], "tắt hết đèn phòng ngủ"),
    ("public_0003", ["contacts", "sms"], "nhắn chị lan mai họp lúc chín giờ"),
]


def catalog_for(app_ids):
    apps = []
    for a in app_ids:
        spec = copy.deepcopy(APPS[a])
        for t in spec["tools"]:
            t["app"] = a
        apps.append({"id": a, **{k: v for k, v in spec.items()}})
    sys_spec = copy.deepcopy(SYS_APP)
    for t in sys_spec["tools"]:
        t["app"] = "sys"
    apps.append({"id": "sys", **sys_spec})
    return {"schema_version": "mock-1", "note": "kind: read | write | meta", "apps": apps}


def installed(app_ids):
    return {a: {"tools": [t["name"] for t in APPS[a]["tools"]]} for a in app_ids}


def state_for(sid, app_ids):
    st = {"_meta": {"question_id": sid, "current_time": NOW,
                    "tools": [t for a in app_ids for t in installed([a])[a]["tools"]]
                    + ["sys.ask_missing_info"]}}
    for a in app_ids:
        st[a] = copy.deepcopy(BASE_STATE[a])
    if "contacts" not in st:
        st["contacts"] = copy.deepcopy(BASE_STATE["contacts"])
    return st


def load_env_cls(split_dir):
    for m in list(sys.modules):
        if m == "zaic_tools" or m.startswith("zaic_tools."):
            del sys.modules[m]
    sys.path.insert(0, str(split_dir))
    try:
        return importlib.import_module("zaic_tools").ToolEnv
    finally:
        sys.path.remove(str(split_dir))


def write_split(out: Path, split: str, rows, app_ids_all, labelled: bool):
    sd = out / split
    if sd.exists():
        shutil.rmtree(sd)
    (sd / "zaic_tools").mkdir(parents=True)
    shutil.copy(HERE / "mock_zaic_tools.py", sd / "zaic_tools" / "__init__.py")
    (sd / f"{split}_tools_catalog.json").write_text(
        json.dumps(catalog_for(app_ids_all), ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        sid, apps = r[0], r[1]
        ed = sd / f"{split}_environment" / sid
        ed.mkdir(parents=True)
        (ed / "state.json").write_text(json.dumps(state_for(sid, apps), ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    Env = load_env_cls(sd)
    lines = []
    for r in rows:
        sid, apps, text = r[0], r[1], r[2]
        item = {"id": sid, "audio": f"audio/{sid}/{sid}.wav",
                "context": {**copy.deepcopy(CONTEXT), "installed_apps": installed(apps)},
                "history": []}
        if labelled:
            gold, hist = r[3], r[4]
            if hist:
                env_h = Env(split, sid, root=str(out))
                u = 0
                for turn in hist:
                    if turn["role"] == "user":
                        u += 1
                        item["history"].append({"role": "user",
                                                "audio": f"audio/{sid}/{sid}_u{u}.wav",
                                                "audio_text": turn["audio_text"]})
                    else:
                        res = [env_h.call(c["name"], c["arguments"]) for c in turn.get("tool_calls", [])]
                        h = {"role": "assistant", "text": turn["text"]}
                        if turn.get("tool_calls"):
                            h["tool_calls"] = turn["tool_calls"]
                            h["tool_results"] = res
                        item["history"].append(h)
                st = env_h.get_state()
                st["_meta"] = json.loads((sd / f"{split}_environment" / sid / "state.json")
                                         .read_text(encoding="utf-8"))["_meta"]
                (sd / f"{split}_environment" / sid / "state.json").write_text(
                    json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
            env = Env(split, sid, root=str(out))
            before = env.get_state()
            results = [env.call(c["name"], c["arguments"]) for c in gold]
            assert all(x.get("ok") for x in results), (sid, results)
            after = env.get_state()
            final_state = {k: v for k, v in after.items() if before.get(k) != v}
            item["audio_text"] = text
            item["label"] = {"tool_calls": gold, "tool_results": results, "final_state": final_state}
        lines.append(json.dumps(item, ensure_ascii=False))
        for rel in [item["audio"]] + [t["audio"] for t in item["history"] if t.get("role") == "user"]:
            write_noise_wav(sd / rel, seed=rel)
    (sd / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_noise_wav(path: Path, seed: str, seconds: float = 1.0, sr: int = 16000):
    import random
    import struct
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n = int(seconds * sr)
    frames = struct.pack(f"<{n}h", *(rng.randint(-800, 800) for _ in range(n)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "mock_data"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_split(out, "train", TRAIN, ["contacts", "sms", "smart_home", "calendar", "banking"], True)
    write_split(out, "public_test", PUBLIC, ["contacts", "sms", "smart_home", "ride"], False)
    print(f"mock data -> {out}")


if __name__ == "__main__":
    main()
