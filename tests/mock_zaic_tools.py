"""Minimal stand-in for the organisers' zaic_tools, enough to exercise the pipeline."""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path


def _fold(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower().strip()


_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
       "array": list, "object": dict}


class ToolEnv:
    def __init__(self, split, sample_id, root="."):
        base = Path(root) / split
        cat = json.loads((base / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
        self._tools = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
        path = base / f"{split}_environment" / sample_id / "state.json"
        self._state = json.loads(path.read_text(encoding="utf-8"))
        self._allowed = set(self._state["_meta"]["tools"])
        self._calls = []

    def validate(self, name, arguments):
        t = self._tools.get(name)
        if t is None or name not in self._allowed:
            return {"ok": False, "error": f"tool không tồn tại hoặc chưa cài: {name}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments phải là object"}
        p = t.get("parameters") or {}
        props = p.get("properties") or {}
        for k in arguments:
            if k not in props:
                return {"ok": False, "error": f"tham số không hợp lệ: {k}"}
        for k in p.get("required") or []:
            if k not in arguments:
                return {"ok": False, "error": f"thiếu tham số bắt buộc: {k}"}
        groups = p.get("x-one-of")
        if groups and not any(all(g in arguments for g in grp) for grp in groups):
            return {"ok": False, "error": "thiếu một trong các nhóm: "
                    + " | ".join(",".join(g) for g in groups)}
        for k, v in arguments.items():
            spec = props[k]
            typ = spec.get("type")
            py = _PY.get(typ)
            bad = py is not None and not isinstance(v, py)
            if typ in ("integer", "number") and isinstance(v, bool):
                bad = True
            if bad:
                return {"ok": False, "error": f"sai kiểu {k}: cần {typ}"}
            if "enum" in spec and v not in spec["enum"]:
                return {"ok": False, "error": f"giá trị {k} không thuộc {spec['enum']}"}
            if "minimum" in spec and isinstance(v, (int, float)) and v < spec["minimum"]:
                return {"ok": False, "error": f"{k} nhỏ hơn {spec['minimum']}"}
            if "maximum" in spec and isinstance(v, (int, float)) and v > spec["maximum"]:
                return {"ok": False, "error": f"{k} lớn hơn {spec['maximum']}"}
        return {"ok": True}

    def call(self, name, arguments):
        self._calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        v = self.validate(name, arguments)
        if not v["ok"]:
            return v
        app, fn = name.split(".", 1)
        handler = getattr(self, f"_h_{app}_{fn}")
        try:
            return handler(**arguments)
        except LookupError as e:     # as zaic_tools: an unknown ID is no error, the tool ran and changed nothing
            return {"ok": True, "done": False, "message": f"không tìm thấy {e}"}

    def get_tool_calls(self):
        return copy.deepcopy(self._calls)

    def get_state(self):
        return copy.deepcopy({k: v for k, v in self._state.items() if k != "_meta"})

    def get_state_hash(self):
        blob = json.dumps(self.get_state(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _contact(self, cid):
        for c in self._state["contacts"]["contacts"]:
            if c["id"] == cid:
                return c
        raise LookupError(cid)

    def _h_contacts_search(self, query=None):
        # as zaic_tools: no query lists every contact
        q = _fold(query or "")
        res = [c for c in self._state["contacts"]["contacts"] if query is None or (q and q in _fold(c["name"]))]
        return {"ok": True, "results": copy.deepcopy(res)}

    def _h_contacts_get(self, contact_id):
        return {"ok": True, "contact": copy.deepcopy(self._contact(contact_id))}

    def _h_sms_send(self, content, contact_id=None, phone_number=None):
        if contact_id is not None:
            to = self._contact(contact_id)["phone"]
        else:
            to = phone_number
        msgs = self._state["sms"]["messages"]
        mid = f"msg_{len(msgs) + 1:03d}"
        msgs.append({"id": mid, "to": to, "content": content})
        return {"ok": True, "message_id": mid}

    def _h_sms_list(self, limit=5):
        return {"ok": True, "messages": copy.deepcopy(self._state["sms"]["messages"][-limit:])}

    def _h_smart_home_get_status(self):
        return {"ok": True, "devices": copy.deepcopy(self._state["smart_home"]["devices"])}

    def _h_smart_home_light_set(self, device_id, power, brightness=None):
        for d in self._state["smart_home"]["devices"]:
            if d["id"] == device_id:
                d["power"] = power
                if brightness is not None:
                    d["brightness"] = brightness
                return {"ok": True, "device": copy.deepcopy(d)}
        raise LookupError(device_id)

    def _h_calendar_create_event(self, title, start_time, end_time=None):
        ev = self._state["calendar"]["events"]
        eid = f"evt_{len(ev) + 1:03d}"
        item = {"id": eid, "title": title, "start_time": start_time}
        if end_time is not None:
            item["end_time"] = end_time
        ev.append(item)
        return {"ok": True, "event_id": eid}

    def _h_calendar_list_events(self, date):
        ev = [e for e in self._state["calendar"]["events"] if e["start_time"].startswith(date)]
        return {"ok": True, "events": copy.deepcopy(ev)}

    def _h_banking_get_balance(self):
        return {"ok": True, "balance": self._state["banking"]["balance"], "currency": "VND"}

    def _h_banking_transfer(self, contact_id, amount, note=None):
        c = self._contact(contact_id)
        b = self._state["banking"]
        if amount > b["balance"]:
            return {"ok": True, "done": False, "message": "số dư không đủ"}
        b["balance"] -= amount
        tid = f"tx_{len(b['transfers']) + 1:03d}"
        item = {"id": tid, "to": c["id"], "amount": amount}
        if note is not None:
            item["note"] = note
        b["transfers"].append(item)
        return {"ok": True, "transfer_id": tid, "balance": b["balance"]}

    def _h_ride_estimate(self, destination):
        return {"ok": True, "destination": destination, "price": 45000}

    def _h_ride_book(self, destination, pickup=None, vehicle="bike"):
        r = self._state["ride"]["bookings"]
        bid = f"ride_{len(r) + 1:03d}"
        r.append({"id": bid, "destination": destination, "pickup": pickup, "vehicle": vehicle})
        return {"ok": True, "booking_id": bid}

    def _h_sys_ask_missing_info(self, tool_name, params):
        return {"ok": True, "asked": {"tool_name": tool_name, "params": list(params)}}
