#!/bin/bash
# Stops the VLM demo started by run_demo.sh — kills the whole tmux session,
# which takes geniex serve, geniex_proxy.py, and start_server.sh down with it.

SESSION="vlm-demo"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Demo stopped (tmux session '$SESSION' killed)."
else
    echo "Demo isn't running (no tmux session '$SESSION')."
fi
