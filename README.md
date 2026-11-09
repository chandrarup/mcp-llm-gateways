# MCP and LLM gateways

Four tasks from a Solution Architect assessment: an MCP server, an MCP security gateway, a streaming PII guardrail, and a rate-limited model router with failover. Everything is Python 3.12 and uses the MCP Python SDK v2, FastAPI, httpx and SQLite.

You need [uv](https://docs.astral.sh/uv/). The demo script also uses `jq`, and the MCP Inspector needs Node.js.

```bash
uv sync
uv run pytest           # 149 tests across all four tasks
./demo.sh               # starts every service on real ports and exercises Tasks 2-4 with curl
```

## Task 1: MCP server

Code is in `task1_mcp_server/`.

```bash
uv run mcp dev task1_mcp_server/server.py      # MCP Inspector
uv run python -m task1_mcp_server.server        # plain stdio
```

The server has two tools, `get_customer_record` and `trigger_refund`, written with `@mcp.tool()`. The SDK builds each input schema from the `Annotated[..., Field(...)]` type hints, and the tools are annotated as read-only and destructive respectively.

Out of the box, `MCPServer` reports bad arguments as a tool result with `isError` set. The task asks for JSON-RPC error codes instead, so a middleware checks every `tools/call` against strict Pydantic models (`extra="forbid"`, strict types) and raises `MCPError(-32602)` with the failing fields. The error lists which fields failed but never repeats the value that was sent.

A request that is valid but fails, such as an unknown customer, takes the other path. It raises `ToolError`, and the model gets a message it can read and retry from.

Logs go to stderr through `logging`. stdout is line-buffered, so a stray `print()` gets flushed while the SDK is still redirecting stdout to stderr and never reaches the protocol stream. A subprocess test injects a `print()` and checks that stdout is still pure JSON-RPC.

The validation tests cover the cases that usually slip through. `\d` matches Unicode digits like `١٢٣٤٥`, so the pattern uses `[0-9]`. The tests also send a trailing newline, a reason made of spaces, NaN and Infinity, `"10"` and `true` in place of numbers, and amounts with more than two decimal places.

## Task 2: MCP security gateway

Code is in `task2_mcp_gateway/`.

```bash
uv run python -m task2_mcp_gateway.auth admin   # mint a demo JWT (or: viewer)
```

Callers authenticate with an HS256 JWT bearer token that must carry `sub`, `role` and `exp`. A missing, expired, forged or `alg=none` token gets HTTP 401.

When a viewer calls a tool whose name starts with `admin_`, the gateway answers with JSON-RPC error `-32001` (Unauthorized Tool Call) under the same `id` and never contacts the downstream server. Names are normalized with NFKC and casefold first, so `Admin_x` or a fullwidth look-alike gets the same treatment.

JSON-RPC batches are rejected outright. Otherwise an admin call could ride along behind a harmless first element, and MCP dropped batching in the 2025-06-18 spec anyway.

Requests that pass go through one pooled `httpx.AsyncClient`. Responses stream back unbuffered, so SSE works, and `Mcp-Session-Id` is preserved. The caller's `Authorization` header is not forwarded downstream.

In production I would replace the prefix check with a per-role allowlist that denies by default, and filter `tools/list` by role.

## Task 3: Streaming PII guardrail

Code is in `task3_llm_guardrail/`.

The gateway exposes an OpenAI-compatible `POST /v1/chat/completions`. It re-streams the upstream SSE and passes every delta through a `StreamRedactor`, with one redactor per choice.

The redactor holds back only text that might still turn into PII:

- the trailing partial word, because an email could still be growing
- a trailing run of digits, spaces and dashes, because card numbers can contain spaces
- any match that touches the end of the buffer, because `a@b.co` might become `a@b.com`

Everything before that goes out immediately. In the demo the first token arrives after about 0.09s, and memory stays bounded by `MAX_BUFFER`.

Text gets split at two levels. TCP can split an SSE line, and `aiter_lines()` puts it back together. A model can split an email across SSE events, and the redactor deals with that.

Card matches must pass a Luhn check, which cuts false positives, and the SSN and card patterns use digit lookarounds so they don't match inside longer numbers. The tests split each sample at every possible 2-way and 3-way position, and also stream it one character at a time. Each result must equal redacting the whole text at once.

Two things are out of scope here: prompts and `tool_calls` arguments are not redacted yet. At higher volume I would look at RE2, Hyperscan or Presidio.

## Task 4: Rate limiter and fallback router

Code is in `task4_router/`.

```bash
curl -XPOST localhost:9200/primary/mode/slow    # ok | rate_limited | slow | error
```

The limit is 50,000 tokens per tenant over a sliding 60-second window. Usage is a log in an on-disk SQLite table, `token_usage(tenant, ts, tokens)`, indexed on `(tenant, ts)`. Tenants are stored as a SHA-256 hash of the API key, never the key itself.

Evicting old rows, summing usage and inserting the new row all happen inside one `BEGIN IMMEDIATE` transaction, with WAL mode and `busy_timeout` enabled. Two concurrent requests therefore can't both see room under the limit. In the concurrency test, 40 threads each ask for 5,000 tokens and exactly 10 get through.

Before calling a model, the gateway reserves an estimate (prompt length / 4, plus `max_tokens`). Afterwards it replaces the estimate with the provider's `usage.total_tokens`, and if no model answered it gives the tokens back. `Retry-After` is the time until enough of the oldest usage expires.

The primary call runs inside a 3000ms `asyncio.timeout`. On timeout the in-flight request is cancelled, so a late reply can't sneak in afterwards. A 429, timeout, 5xx or network error sends the request to the secondary provider. Any other 4xx does not fail over, since the backup would reject the same request.

Every error comes back as `{"error": {"type", "message", "request_id"}}`. Upstream response bodies, model names, hostnames and stack traces only appear in the server logs.

To run several gateway instances I would move the limiter to Redis, using a sorted set or a Lua script. I would also add a circuit breaker on the primary and count tokens with a real tokenizer such as `tiktoken`.
