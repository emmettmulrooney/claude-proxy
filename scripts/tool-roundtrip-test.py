"""Simulates one Agent-mode tool loop through the proxy, the way Cursor sends it.

Usage: PROXY_KEY=... python3 scripts/tool-roundtrip-test.py https://your-proxy.up.railway.app
"""
import json
import os
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
KEY = os.environ["PROXY_KEY"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the user's project.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def stream(messages):
    body = {
        "model": "proxy-opus",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": TOOLS,
        "tool_choice": "auto",
        "messages": messages,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "content-type": "application/json"},
    )
    text, calls, finish = "", {}, None
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:500])
        sys.exit(1)
    raw = resp.read().decode()
    for line in raw.splitlines():
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        if "choices" not in d and "type" in d:
            print("error event:", line[:500])
            sys.exit(1)
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"id": None, "name": "", "args": ""})
                c["id"] = tc.get("id") or c["id"]
                c["name"] += (tc.get("function") or {}).get("name") or ""
                c["args"] += (tc.get("function") or {}).get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    if not text and not calls:
        print("unexpected response:", raw[:500])
        sys.exit(1)
    return text, list(calls.values()), finish


messages = [
    {"role": "system", "content": "You are a coding agent. Use tools to look at files."},
    {"role": "user", "content": "Read package.json and tell me the project's name. Use the read_file tool."},
]
text, calls, finish = stream(messages)
print("turn 1 finish_reason:", finish)
for c in calls:
    print(f"turn 1 tool call: {c['name']}({c['args']})")
if not calls:
    print("no tool call; reply was:", text)
    sys.exit(1)

messages.append(
    {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
            for c in calls
        ],
    }
)
for c in calls:
    messages.append(
        {"role": "tool", "tool_call_id": c["id"], "content": '{"name": "claude-proxy", "version": "1.0.0"}'}
    )

text, calls, finish = stream(messages)
print("turn 2 finish_reason:", finish)
print("turn 2 reply:", text.strip())
