"""Spawns the real server and speaks raw JSON-RPC over stdio to prove stdout carries only protocol frames."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Adds a tool that print()s, to prove a stray print can't corrupt the protocol stream.
NOISY_SERVER = """
import sys
import task1_mcp_server.server as s

@s.mcp.tool()
def noisy() -> str:
    \"\"\"Prints to stdout.\"\"\"
    print("stray debug print")
    return "ok"

s.INPUT_MODELS["noisy"] = s.StrictInput
sys.stdout.reconfigure(line_buffering=True)
s.mcp.run()
"""

MESSAGES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-12345"}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "trigger_refund", "arguments": {"customer_id": "BAD", "amount": -1, "reason": "x"}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "noisy", "arguments": {}}},
]


def run_session(args: list[str]) -> tuple[list[dict], str]:
    proc = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    assert proc.stdin and proc.stdout
    responses = []
    for msg in MESSAGES:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        if "id" in msg:
            responses.append(json.loads(proc.stdout.readline()))
    rest, stderr = proc.communicate(timeout=10)
    for line in filter(str.strip, rest.splitlines()):
        responses.append(json.loads(line))  # anything left on stdout must still be JSON-RPC
    return responses, stderr


def test_stdio_session_is_pure_jsonrpc_and_logs_go_to_stderr():
    responses, stderr = run_session(["-c", NOISY_SERVER])
    by_id = {r["id"]: r for r in responses}

    assert all(r["jsonrpc"] == "2.0" for r in responses)
    assert {t["name"] for t in by_id[2]["result"]["tools"]} >= {"get_customer_record", "trigger_refund"}
    assert by_id[3]["result"]["structuredContent"]["name"] == "Alan Turing"
    assert by_id[4]["error"]["code"] == -32602
    assert by_id[5]["result"]["isError"] is False

    assert "stray debug print" in stderr
    assert "lookup customer_id=CUST-12345" in stderr
