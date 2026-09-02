#!/bin/bash
# Launches the three long-running processes needed for the offline VLM demo
# (geniex serve -> geniex_proxy.py -> start_server.sh) as tmux windows in a
# single session, so nothing dies with the terminal/SSH session. Manual
# per-terminal commands are documented in demo_documentation.md; this script
# just automates typing them in and waiting for each one to be ready before
# starting the next, same as a human driving three terminals would.
#
# Optional first argument: the path to open in the browser once ready
# (default "/", the main demo). scripts/run_traffic_demo.sh is a thin
# wrapper that calls this with "/traffic" - both start the SAME single
# server process (it serves both pages), just landing on a different page,
# since /traffic needs the same GenieX chain running underneath anyway.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="vlm-demo"
MODEL="qualcomm/Qwen3-VL-4B-Instruct"
UI_PATH="${1:-/}"
UI_URL="https://localhost:8090${UI_PATH}"

cd "$REPO_DIR"

# Waits for the webui to actually accept HTTPS, then opens it in Firefox.
# Runs detached from the tmux session so it doesn't block attaching below.
open_browser_when_ready() {
    until curl -sfk "$UI_URL" >/dev/null 2>&1; do
        sleep 2
    done
    firefox --new-window "$UI_URL" >/dev/null 2>&1 &
    disown
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Demo session '$SESSION' is already running — attaching."
    open_browser_when_ready &
    disown
    exec tmux attach -t "$SESSION"
fi

echo "Starting demo in tmux session '$SESSION'..."

# Window 1: geniex serve (NPU inference server)
tmux new-session -d -s "$SESSION" -n geniex "geniex serve"

# Window 2: proxy — waits for geniex's port before starting
tmux new-window -t "$SESSION" -n proxy \
    "until curl -sf http://127.0.0.1:18181/v1/models >/dev/null 2>&1; do sleep 2; done; python3 scripts/geniex_proxy.py"

# Window 3: web UI — waits for the proxy's port before starting
tmux new-window -t "$SESSION" -n webui \
    "until curl -sf http://127.0.0.1:18182/v1/models >/dev/null 2>&1; do sleep 2; done; ./scripts/start_server.sh --api-base http://127.0.0.1:18182/v1 --model $MODEL"

tmux select-window -t "$SESSION:geniex"

open_browser_when_ready &
disown

echo "geniex serve can take a while to load the model — proxy and webui will wait for it automatically."
echo "Firefox will open $UI_URL automatically once the webui is up (accept the self-signed cert warning)."
echo "Switch windows with Ctrl+B then window number (0=geniex, 1=proxy, 2=webui)."
echo "Detach any time with Ctrl+B then D — the demo keeps running. Reattach with: tmux attach -t $SESSION"

exec tmux attach -t "$SESSION"
