# Advantech Live VLM Demo — Setup & Deployment

Full step-by-step for bringing up the offline, on-device VLM demo on an
Advantech AIR-055 (Qualcomm QCS9075 / IQ-9 Dragonwing, 100 TOPS variant),
running Qwen3-VL-4B-Instruct via GenieX on the Hexagon NPU. Covers a fresh
device from scratch and what to check when things aren't working.

See `CLAUDE.md` for architecture notes, known upstream quirks, and things
deliberately *not* to change. This doc is the operational how-to; `CLAUDE.md`
is the why.

---

## Architecture

```
Browser (WebRTC) <--> live-vlm-webui server.py <--> geniex_proxy.py (18182) <--> geniex serve (18181, NPU)
```

Three long-running processes, three terminals, all local — no internet
required once the model is downloaded.

---

## 1. NPU / GenieX setup (device-level, one-time)

Run directly on the AIR-055 (Linux ARM64), no sudo required:

```bash
curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
which geniex && geniex --version
```

Pull the model (pre-compiled QAIRT bundle, ~4.1 GiB, pre-quantized `W4A16` —
no precision choice needed, and this is the **only** download in the whole
setup that needs internet access):

```bash
geniex pull ai-hub-models/Qwen3-VL-4B-Instruct
geniex list   # confirm it's cached
```

### Sanity-check GenieX directly, before touching any UI

```bash
geniex serve
```

In a second terminal:

```bash
curl http://127.0.0.1:18181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qualcomm/Qwen3-VL-4B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

> **Known GenieX bug:** use the **bare** model name here
> (`qualcomm/Qwen3-VL-4B-Instruct`), never the tagged form
> (`qualcomm/Qwen3-VL-4B-Instruct:W4A16`) that `geniex list` and `/v1/models`
> display — the tagged form fails with
> `SDKError(Invalid input parameters or handle), quantization 'W4A16' not found`.
> `scripts/geniex_proxy.py` (step 3 below) exists specifically to work around
> this so no frontend has to special-case it.

### Confirm inference is actually running on the NPU

```bash
export GENIEX_LOG=INFO   # set before starting geniex serve
```

Send a request and look for `Found device: HTP0` in the log (`HTP` = Hexagon
Tensor Processor). Supporting evidence: the startup log links
`libcdsprpc.so`/`libadsprpc.so` (Qualcomm's fastRPC transport to the DSP), and
`htop` during inference should **not** show sustained full-core CPU
saturation — the heavy compute is off-CPU. Qualcomm AI Hub / `qairt` models
only support NPU; `--compute cpu`/`--compute gpu` are rejected by design.

### Keeping `geniex serve` running reliably

It has been observed to stop unexpectedly between sessions (e.g. an SSH
session closing). Run it under `tmux` for anything beyond a quick test:

```bash
tmux new -s geniex
geniex serve
# Ctrl+B then D to detach — keeps running in the background
# reconnect later with: tmux attach -t geniex
```

If a request ever hangs or fails, the first check is always:

```bash
curl http://127.0.0.1:18181/v1/models
```

If that fails to connect, `geniex serve` has stopped and needs restarting.

---

## 2. Get this fork onto the device

This demo runs from **this git fork** (Advantech branding, redesigned UI,
GenieX proxy fix, prompt/timeout tuning) — not the vanilla upstream
`live-vlm-webui` PyPI package, which has none of that.

```bash
git clone https://github.com/mdfaiz98/live-vlm-webui.git
cd live-vlm-webui
git checkout design-v3   # the current working branch — check `git branch -a` for the latest
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

`scripts/geniex_proxy.py` and all Advantech branding assets
(`src/live_vlm_webui/static/images/`) are already tracked in git — cloning
the branch brings everything over, nothing to copy by hand.

---

## 3. Run it (three terminals)

```bash
# Terminal 1 — NPU inference server (tmux recommended, see above)
geniex serve

# Terminal 2 — model-tag workaround proxy
python3 scripts/geniex_proxy.py

# Terminal 3 — the web UI itself
./scripts/start_server.sh --api-base http://127.0.0.1:18182/v1 --model qualcomm/Qwen3-VL-4B-Instruct
```

`start_server.sh` auto-detects/activates `.venv`, generates a self-signed
`cert.pem`/`key.pem` on first run if missing, and serves on
`https://0.0.0.0:8090`.

Open **`https://localhost:8090`** in a browser on the device (or
`https://<device-ip>:8090` from another machine on the same LAN). Accept the
self-signed certificate warning (Advanced → Proceed).

Always point every client at **port 18182** (the proxy), never `18181`
directly — that's the whole reason the proxy exists.

---

## 4. Known upstream quirks (not our bugs — GenieX developer preview software)

See `CLAUDE.md` for the full list and reasoning; summary:

- **Model-tag mismatch** — worked around by `geniex_proxy.py` (§1 above).
- **Occasional very slow response** (2-6+ min instead of 2-4s) — thermal
  throttling / NPU contention / fastRPC allocation hiccups, not yet
  root-caused. `vlm_service.py` has a 25s client timeout **plus** a 40s
  cooldown after any failure — the cooldown prevents request pile-up
  (GenieX keeps processing abandoned requests otherwise, causing cascading
  delays and `broken pipe` errors). Don't remove it without a replacement.
- **Gemma GGUF models fail on the Hexagon NPU** (`fastrpc_mmap failed`) —
  works on `--compute cpu` only, not relevant to the Qwen3-VL-4B path used
  here.
- **Qwen3-VL-8B-Instruct** is listed on Qualcomm AI Hub's website but
  `geniex pull ai-hub-models/Qwen3-VL-8B-Instruct` currently returns "not
  found" — catalog is out of sync with what's actually pull-able; recheck
  periodically if a bigger model is wanted.
- **The video feed depends on the physical network interface being up**
  (aioice, the WebRTC library in use, cannot gather loopback candidates —
  confirmed dead end, do not try to force loopback-only WebRTC). Unplugging
  the network cable will freeze/blank the live video even though VLM
  inference keeps working (that path is genuine `127.0.0.1` loopback HTTP,
  unaffected). No fix deployed yet; see project chat history for the
  options considered (moving video display off WebRTC onto a plain
  HTTP/WebSocket frame stream) if this needs solving for a given site.

---

## 5. Troubleshooting checklist

| Symptom | Check |
|---|---|
| UI won't load | Is `start_server.sh` actually running? Check terminal 3 for errors. |
| "Connecting..." never resolves / model errors | Is `geniex_proxy.py` (terminal 2) running, and is the UI's API Base URL `http://127.0.0.1:18182/v1` (not 18181)? |
| Proxy itself failing | `curl http://127.0.0.1:18181/v1/models` — if this fails, `geniex serve` (terminal 1) has died; restart it. |
| Video feed frozen but text responses still updating | Expected if the network cable was unplugged — see the WebRTC quirk above. Reconnect the cable; the frontend auto-retries every 3s. |
| A single response takes minutes | Known GenieX stall (see quirks above) — the 40s cooldown will kick in automatically after the 25s timeout; wait it out rather than restarting `geniex serve` mid-demo. |
| Port 8090 already in use | `./scripts/start_server.sh` will report this and suggest `lsof -ti :8090` / a different `--port`. |

---

## 6. Deploying to another IQ-9 device — quick checklist

1. `geniex pull ai-hub-models/Qwen3-VL-4B-Instruct` (needs internet once; or
   copy the existing device's GenieX model cache directly if internet is
   unreliable on the new site).
2. `git clone` this repo, `git checkout design-v3` (confirm this is still
   the latest branch — check `git branch -a` / ask whoever's driving the
   repo if unsure).
3. `python3 -m venv .venv && pip install -e .`
4. Three terminals: `geniex serve` → `scripts/geniex_proxy.py` →
   `./scripts/start_server.sh --api-base http://127.0.0.1:18182/v1 --model qualcomm/Qwen3-VL-4B-Instruct`.
5. Sanity-check with the `curl` command in §1 before opening the browser.
