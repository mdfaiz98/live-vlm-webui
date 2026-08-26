#!/bin/bash
# Generates the "VLM Demo" / "VLM Demo (Stop)" desktop launcher icons for
# THIS device, from the templates in scripts/desktop/. Run once per device
# after cloning — the templates use a __REPO_DIR__ placeholder so the icons
# work regardless of which user/path the repo was cloned into.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="$REPO_DIR/scripts/desktop"
DESKTOP_DIR="$HOME/Desktop"

if [ ! -d "$DESKTOP_DIR" ]; then
    echo "No ~/Desktop directory found ($DESKTOP_DIR) — creating it."
    mkdir -p "$DESKTOP_DIR"
fi

for tmpl in "$TEMPLATE_DIR"/*.desktop.tmpl; do
    name="$(basename "$tmpl" .tmpl)"
    dest="$DESKTOP_DIR/$name"
    sed "s|__REPO_DIR__|$REPO_DIR|g" "$tmpl" > "$dest"
    chmod +x "$dest"
    # GNOME/Nautilus refuses to run untrusted .desktop files by default.
    if command -v gio >/dev/null 2>&1; then
        gio set "$dest" "metadata::trusted" true 2>/dev/null || true
    fi
    echo "Installed $dest"
done

echo "Done. Icons should now be double-clickable from the desktop."
