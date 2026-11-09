"""A minimal downstream MCP server speaking JSON-RPC over HTTP POST.

Run: uv run uvicorn task2_mcp_gateway.mock_downstream:app --port 9000
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

TOOLS = [
    {"name": "get_status", "description": "Service health", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object", "properties": {}}},
]

app = FastAPI(title="mock-downstream-mcp")
app.state.received = []  # every request that reached the downstream, for tests and demos


def ok(msg_id: Any, result: dict) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": result})


@app.post("/mcp")
async def handle(request: Request) -> Response:
    msg = await request.json()
    app.state.received.append({"body": msg, "headers": dict(request.headers)})
    method, msg_id = msg.get("method"), msg.get("id")

    if msg_id is None:
        return Response(status_code=202)  # notification: acknowledged, no body
    if method == "initialize":
        return ok(msg_id, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "mock-downstream", "version": "1.0"}})
    if method == "tools/list":
        return ok(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        name = msg["params"]["name"]
        text = "key rotated" if name == "admin_reset_key" else "all systems nominal"
        return ok(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}})
