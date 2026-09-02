#!/bin/bash
# Starts the same demo stack as run_demo.sh (geniex serve -> proxy ->
# start_server.sh), but opens straight to the traffic monitoring cascade
# demo (/traffic) instead of the main page - a dedicated command for
# showing that specific capability, without needing to remember to
# navigate over manually. Same single server process either way; see
# demo_documentation.md §7 for what /traffic needs (model export, demo
# video) before this will show anything useful.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_DIR/scripts/run_demo.sh" /traffic
