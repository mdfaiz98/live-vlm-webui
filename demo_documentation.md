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

There's also a second, separate demo page at **`https://localhost:8090/traffic`**
(object detection → click-to-caption cascade) running in this same process —
see §7 for its one-time setup before it'll work.

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

   Or, instead of three manual terminals: `./scripts/run_demo.sh` starts all
   three in a single `tmux` session (waiting for each dependency's port
   before starting the next) and opens Firefox once the webui is ready.
   Stop everything with `./scripts/stop_demo.sh`.
5. Optional — double-clickable desktop icons for the two scripts above:
   `./scripts/install_desktop_icons.sh` (run once per device; generates
   `~/Desktop/VLM-Demo.desktop` and `~/Desktop/VLM-Demo-Stop.desktop` from
   the templates in `scripts/desktop/`, pointed at wherever this device
   cloned the repo).
6. Sanity-check with the `curl` command in §1 before opening the browser.

---

## 7. Optional: Traffic Monitoring Cascade Demo (`/traffic`)

A second, separate demo page — object detection running continuously on
the Hexagon NPU (YOLO26n), feeding cropped vehicle images into the same
GenieX/VLM pipeline for on-demand descriptions. Deliberately kept as its
own route (`/traffic`) with its own pipeline, not merged into the main
demo at `/` — see `CLAUDE.md` and the architecture plan doc for why.

It runs inside the **same** `start_server.sh` process as the main demo —
no extra terminal needed. Once terminal 3 (§3 above) is up, just navigate
to `https://localhost:8090/traffic` (or `/traffic` on whatever host/port
the main demo is running on).

### 7.1 One-time setup (not covered by §1–3 above)

Two things this page needs that the main demo doesn't: a detection model,
and a demo video. Neither is committed to git (see `.gitignore` —
`models/` and `demo_assets/` are both excluded; the model is a ~10 MB
binary regenerable from a public model zoo, and the video is a licensed/
provided clip, not something to bundle into every clone of this repo).

**Install the extra Python dependency:**

```bash
source .venv/bin/activate
pip install -e ".[traffic]"   # adds ai-edge-litert (TFLite runtime for the NPU delegate)
```

**Export the YOLO26n detection model**, via Qualcomm AI Hub (needs a free
Qualcomm ID + API token, and internet access for this one-time step — the
board itself doesn't need internet afterward, same as the GenieX model
pull in §1):

```bash
python3.12 -m venv ~/qai-hub-env
source ~/qai-hub-env/bin/activate
pip install "qai-hub-models[yolo26-det]"

qai-hub configure --api_token YOUR_TOKEN   # get this from your account at aihub.qualcomm.com
qai-hub list-devices | grep -i 9075        # confirm "Dragonwing IQ-9075 EVK" is listed

mkdir -p /path/to/live-vlm-webui/models
qai-hub-models export yolo26_det --device "Dragonwing IQ-9075 EVK" \
    --runtime tflite --precision float --ckpt-name yolo26n.pt \
    --skip-profiling --skip-inferencing --output-dir /path/to/live-vlm-webui/models/

cd /path/to/live-vlm-webui/models/
mv yolo26_det-tflite-float/yolo26_det.tflite yolo26n_det_qcs9075.tflite
mv yolo26_det-tflite-float/labels.txt coco_labels.txt
rmdir yolo26_det-tflite-float
```

`models/` should now contain `yolo26n_det_qcs9075.tflite` and
`coco_labels.txt` — both required; the server logs a clear warning and
disables `/traffic` (without affecting the main demo) if either is
missing at startup.

> **Why float, not quantized:** float already runs at 60+ FPS on this
> NPU (see §7.4) with full confidence-score resolution — no need to fight
> quantization for a nano-sized model. YOLO26 doesn't even offer a w8a8
> variant on AI Hub, only float and w8a16.

**Place the demo video:**

```bash
mkdir -p /path/to/live-vlm-webui/demo_assets
cp /path/to/your/highway-footage.mp4 /path/to/live-vlm-webui/demo_assets/car-highway.mp4
```

Any clip works — the current one is a 1920x1080, 30fps, 15-second highway
loop. It plays on repeat automatically (`VideoFileTrack(loop=True)`); the
filename `car-highway.mp4` is currently hardcoded in `server.py`
(`TRAFFIC_VIDEO_PATH`) rather than configurable via CLI flag.

### 7.2 What's on the page

- Live bounding boxes burned into the video itself (server-side), with
  label + confidence — not a client-side canvas overlay.
- **Detected Vehicles** panel — each physical vehicle logged once (a
  lightweight IoU tracker dedupes per-frame detections; see `tracker.py`),
  using its largest-observed crop, newest first, capped at 40 shown (the
  counter badge reflects the true running total, not just what's visible).
- Click a card to open a detail view (larger image) with a **Describe**
  button — sends that vehicle's crop through GenieX for a 1-2 sentence
  description (color, type, brand; license plate only if clearly legible,
  never guessed at). One caption in flight at a time, same 40s
  failure-cooldown pattern as `vlm_service.py` (§4) to protect against
  GenieX pile-up.
- **✎ Prompt** — edit the caption prompt live; applies to future
  Describe clicks, shared across reconnects (server-side, resets on
  server restart), with a one-click reset to default.
- Stop/Start (halts the whole pipeline — no new detections logged while
  stopped) and a separate pause/play on the video itself (freezes the
  actual server-side video position, not just the local screen — resuming
  continues from exactly where it was paused, not "now").
- Live FPS counter (top-left of the video).
- **Local System Stats** card along the bottom — the same card as the main
  demo's (CPU, RAM, Thermal), plus an **Accelerators** card ported from the
  QCS6490 drone detection demo (`QualcommMonitor` in `gpu_monitor.py`).
  Where the accelerator numbers come from:
  - **GPU** — Adreno busy time summed over all DRM clients (desktop,
    browser, this app) from `/proc/*/fdinfo` `drm-engine-gpu`, plus the
    devfreq clock. Mostly ~1% here: the video is CPU-decoded and nothing in
    the cascade uses the GPU.
  - **NPU · AI models** — one ring for both NPU workloads: the time the
    YOLO26n detector spends in its Hexagon `invoke()` plus the time a
    Describe request is in flight on GenieX (Qwen3-VL, in the GenieX
    process), capped at 100%. ~30% with detection running, 100% while a
    description is being written. The NPU has no readable utilization
    counter, so this is measured from the app, not read from hardware.
  The main demo at `/` gets the same data source (its header now shows
  "Advantech AFE-A503 (IQ9)"), but only renders CPU/RAM/thermal as before.
- **⚙ Settings** drawer (header): **Video** — the files in `demo_assets/`
  (*Highway traffic* = `car-highway.mp4`, *Parking entrance (barrier)* =
  `parking-entrance-barrier.mp4`; names in `TRAFFIC_VIDEO_NAMES`), looped,
  applied with *Play this video*; **Detection model** — every `.tflite` in
  `models/` (switches live); **Vision-language model** — whatever GenieX
  serves (used for Describe). Plus a **theme toggle** (Auto / Light / Dark),
  shared with the main demo's saved preference.
- **Entry zone** (parking video): only vehicles whose box bottom-centre is
  inside the amber *ENTRY ZONE* (the ground in front of the barrier) are
  boxed and logged, so street traffic behind the barrier and the parked car
  at the right edge are ignored. Per-video zones live in
  `TRAFFIC_ENTRY_ZONES` in `server.py` (fractions of the frame); videos
  without one use the whole frame.
- **Parking demo video**: stitched from nine clips of the [AGH University
  parking database](https://qoe.agh.edu.pl/parking-database/) (720p,
  upscaled to 1080p) — offered "to the research community free of charge",
  so treat it as research footage, not ours. Rebuild: download
  `agh_src{13,15,18,20,10,1,11,12,21}_hrc0.avi` and join them with
  ffmpeg's `concat` filter (the concat *demuxer* fails on these AVIs), then
  keep the first 1:37 (`ffmpeg -i full.mp4 -t 97 -c:v libx264 -crf 20 -an
  parking-entrance-barrier.mp4`): the five cars *entering* the garage, each
  logged with a clear plate. The full 3:03 cut (with cars leaving) is kept on
  the board at `~/Videos/parking-extras/`. When an entry-zone video loops,
  the tracker is cleared (`VideoFileTrack.on_loop`), so the first car of the
  next pass isn't mistaken for the last car of the previous one.
- Describe uses the vehicle's latest best crop (the tracker keeps refining
  it after the card appears), and the default prompt puts the plate on its
  own line with an explicit "Plate: not readable" — see the comment on
  `TRAFFIC_CAPTION_DEFAULT_PROMPT`.
- The page is sized to fit on one screen without scrolling (e.g. 1920x1080
  full screen, via the ⤢ button in the header): the video shrinks slightly
  on shorter screens so the stats card stays visible. If the stats card's
  height ever changes, re-measure `--fit-reserve` in `traffic.css`.

### 7.3 Known quirks specific to `/traffic`

- **A VPN (e.g. Tailscale) active in the browser can cause repeated
  ~3s disconnects.** The VPN's own network interface gets offered up as a
  WebRTC ICE candidate alongside the real ones, and something about that
  destabilizes the connection. If `/traffic` keeps reconnecting in a
  loop, check for and disable any active VPN first — this was the root
  cause the one time it was fully diagnosed, not a code bug.
- **A rare native crash** (`munmap_chunk(): invalid pointer` / `free():
  invalid pointer` / SIGSEGV) can still occur under heavy reconnect churn,
  traced to a PyAV/libav thread-safety issue when video decode contexts
  are opened/closed in rapid succession — mitigated (server-side rate
  limiting on new connections, `buffered=False` on the frame relay so a
  processing backlog can't build up) but not fully eliminated. If the
  page shows "Disconnected" and won't recover, check whether the whole
  `start_server.sh` process died (`ps aux | grep live_vlm_webui.server`)
  and restart it if so — the main demo at `/` is unaffected either way,
  since a `/traffic` crash takes down the shared process, but the reverse
  isn't true in normal operation (this is a `/traffic`-specific failure
  mode, not something the main demo's own WebRTC path triggers).
- Only **one active `/traffic` connection at a time** — opening it in a
  second tab/browser cleanly closes the first (by design, not a bug; see
  `traffic_active_pc` in `server.py`).

### 7.4 Reference numbers (this exact model/board/clip)

| Stage | Measured |
|---|---|
| Detection latency (float model, HTP delegate) | ~13-16 ms/frame steady-state (~30ms first frame, incl. one-time delegate warmup) |
| End-to-end throughput (detection + draw + encode) | ~28 fps on the full 1920x1080 clip |
| VLM caption latency | Same as main demo (~2-4s typical; same GenieX stall risk applies) |
