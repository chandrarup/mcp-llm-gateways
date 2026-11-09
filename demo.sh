#!/usr/bin/env bash
# Starts every service on real ports and exercises Tasks 2-4 with curl.
# Usage: ./demo.sh
set -uo pipefail
cd "$(dirname "$0")"

export RATE_LIMIT_DB=/tmp/sa_demo_rate_limits.sqlite3
rm -f "$RATE_LIMIT_DB" "$RATE_LIMIT_DB"-wal "$RATE_LIMIT_DB"-shm

cleanup() { pkill -f "uvicorn task[234]_" 2>/dev/null; }
trap cleanup EXIT

start() { uv run uvicorn "$1" --port "$2" --log-level warning & }
start task2_mcp_gateway.mock_downstream:app 9000
start task2_mcp_gateway.gateway:app         8000
start task3_llm_guardrail.mock_llm:app      9100
start task3_llm_guardrail.gateway:app       8100
start task4_router.mock_providers:app       9200
start task4_router.app:app                  8200
sleep 4

section() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
JSON='content-type: application/json'
ask() {  # ask <tenant-key> [max_tokens] -> prints status, provider, body
  curl -s -D /tmp/sa_h -o /tmp/sa_b -w '%{http_code}' localhost:8200/v1/chat/completions \
    -H "Authorization: Bearer $1" -H "$JSON" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":${2:-256}}"
}

# ---------------------------------------------------------------- Task 2
VIEWER=$(uv run python -m task2_mcp_gateway.auth viewer)
ADMIN=$(uv run python -m task2_mcp_gateway.auth admin)
mcp() { curl -s -w '  [HTTP %{http_code}]' localhost:8000/mcp -H "$JSON" ${1:+-H "Authorization: Bearer $1"} -d "$2"; echo; }

section "Task 2: viewer tools/list (forwarded)"
mcp "$VIEWER" '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
section "Task 2: viewer calls admin_reset_key (blocked at gateway)"
mcp "$VIEWER" '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key"}}'
section "Task 2: admin calls admin_reset_key (forwarded)"
mcp "$ADMIN" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key"}}'
section "Task 2: no token"
mcp "" '{"jsonrpc":"2.0","id":4,"method":"tools/list"}'

# ---------------------------------------------------------------- Task 3
section "Task 3: streamed completion, PII redacted live (seconds since start | event)"
START=$(python3 -c 'import time; print(time.time())')
curl -sN localhost:8100/v1/chat/completions -H "$JSON" \
  -d '{"messages":[{"role":"user","content":"who is the customer?"}],"stream":true}' |
while IFS= read -r line; do
  [ -z "$line" ] && continue
  python3 - "$START" "$line" <<'PY'
import json, sys, time
start, line = float(sys.argv[1]), sys.argv[2]
data = line.removeprefix("data: ")
text = data if data == "[DONE]" else repr(json.loads(data)["choices"][0]["delta"].get("content", ""))
print(f"{time.time() - start:5.2f}s | {text}")
PY
done

# ---------------------------------------------------------------- Task 4
section "Task 4: primary healthy"
echo "HTTP $(ask tenant-a-key) via $(grep -i x-gateway-provider /tmp/sa_h | cut -d' ' -f2 | tr -d '\r'): $(jq -c .choices[0].message.content /tmp/sa_b)"

curl -s -XPOST localhost:9200/primary/mode/rate_limited >/dev/null
section "Task 4: primary returns 429 -> failover"
echo "HTTP $(ask tenant-a-key) via $(grep -i x-gateway-provider /tmp/sa_h | cut -d' ' -f2 | tr -d '\r')"

curl -s -XPOST localhost:9200/primary/mode/slow >/dev/null
section "Task 4: primary hangs -> failover after the 3s deadline"
T0=$(python3 -c 'import time; print(time.time())')
CODE=$(ask tenant-a-key)
echo "HTTP $CODE via $(grep -i x-gateway-provider /tmp/sa_h | cut -d' ' -f2 | tr -d '\r') after $(python3 -c "import time; print(f'{time.time()-$T0:.2f}s')")"

curl -s -XPOST localhost:9200/primary/mode/error >/dev/null
section "Task 4: primary 500 with a leaky traceback -> failover, nothing leaked"
echo "HTTP $(ask tenant-a-key) via $(grep -i x-gateway-provider /tmp/sa_h | cut -d' ' -f2 | tr -d '\r')"

section "Task 4: unknown API key"
echo "HTTP $(ask bad-key): $(cat /tmp/sa_b)"

curl -s -XPOST localhost:9200/primary/mode/ok >/dev/null
section "Task 4: tenant-b burns its 50k tokens/min budget (8000 max_tokens per call)"
for _ in $(seq 1 8); do printf '%s ' "$(ask tenant-b-key 8000)"; done; echo
echo "last response: $(cat /tmp/sa_b)"
echo "retry-after:   $(grep -i retry-after /tmp/sa_h | cut -d' ' -f2 | tr -d '\r')s"
section "Task 4: tenant-a is unaffected by tenant-b's limit"
echo "HTTP $(ask tenant-a-key)"
