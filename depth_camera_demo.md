# Depth Camera Demo (Orbbec Gemini 2L) — branch `depthCameraTest`

Adds live depth-camera support (Orbbec Gemini 2L) to the Advantech Live VLM
demo, on a **separate page** (`/depth`) that runs alongside the existing
webcam/RTSP/file demo — nothing about the original demo changed or was
touched. This doc explains what's different and why, so it's clear what
you're looking at when picking this branch back up later.

**Status:** experimental, lives only on `depthCameraTest`. Not merged into
`main`. Earmarked for the Electronica demo — see `CLAUDE.md`/git log for
what's on `main` at any given point if this branch needs rebasing before
that.

---

## 1. What's new, in one sentence

A second page, `/depth`, that shows Color + Depth + IR feeds from an Orbbec
Gemini 2L side by side with the same VLM output pipeline the main page uses,
plus live system/camera telemetry and a prompt editor — built as a
self-contained addition, not a modification of the existing demo.

## 2. Why a separate page instead of adding Orbbec as a source on the main page

The main page's video-source selector (webcam / RTSP / local file) is
existing, working, demoed functionality. Rather than risk it, Orbbec support
was built as an entirely new page (`depth.html` / `depth.css` / `depth.js`)
with its own backend routes. The main page is completely unaware the Orbbec
code exists — `git diff main` on `src/live_vlm_webui/static/index.html` and
`app.js` is empty.

## 3. Architecture

```
Orbbec Gemini 2L (USB3.0)
        |
   pyorbbecsdk2 (Python bindings)
        |
   orbbec_source.py — OrbbecCameraManager (singleton, owns the device)
        |                                   |
   Color frames                    Depth/IR frames + IMU telemetry
        |                                   |
   OrbbecColorTrack                  encode_depth_jpeg() / encode_ir_jpeg()
   (aiortc VideoStreamTrack)         / get_telemetry()
        |                                   |
   existing WebRTC + VLM        dedicated WebSocket: /ws/orbbec
   pipeline (/offer,                 (~12fps, base64 JPEG + JSON)
   VideoProcessorTrack,                     |
   GenieX) — same code                browser: <img> elements,
   path as webcam/RTSP/file           updated on message
        |                                   |
        +---------------- browser: /depth page ----------------+
```

**Why Depth/IR don't ride WebRTC too:** `CLAUDE.md` documents that `aioice`
(the WebRTC library in use) hard-excludes the loopback interface from
candidate gathering with no override, and is generally fragile. Rather than
add two more WebRTC tracks, Depth and IR are pushed as JPEG frames over a
plain WebSocket. Cheaper to encode (single image vs. video codec), no
jitter-buffer latency, and doesn't touch the aioice-sensitive code path at
all. Color still uses WebRTC because that's what lets it reuse the *existing*
VLM capture pipeline (`VideoProcessorTrack`) unchanged.

**Why the camera manager is a singleton:** the physical USB device can only
be opened by one process/pipeline at a time (confirmed the hard way — see
§6). `get_orbbec_manager()` returns one shared instance; the WebRTC Color
track and the `/ws/orbbec` WebSocket both read from it rather than each
opening their own connection to the camera.

## 4. New files

| File | Purpose |
|---|---|
| `src/live_vlm_webui/orbbec_source.py` | `OrbbecCameraManager` (owns the SDK pipelines, caches latest frames/telemetry behind a lock) + `OrbbecColorTrack` (aiortc track wrapping the manager) |
| `src/live_vlm_webui/static/depth.html` | The `/depth` page markup |
| `src/live_vlm_webui/static/depth.css` | Page-specific styles — reuses design tokens/component classes from `style.css` (ring gauges, thermal chart, form fields, `.result-*` typography) rather than duplicating them |
| `src/live_vlm_webui/static/depth.js` | Page logic — WebRTC connect, `/ws/orbbec` client, prompt editor, telemetry rendering |
| `depth_camera_demo.md` | This file |

## 5. Changes to existing files

- **`src/live_vlm_webui/server.py`**:
  - `offer()` gained an `elif use_orbbec:` branch (alongside the existing
    RTSP/file/webcam branches) — creates `OrbbecColorTrack`, wraps it in the
    same `VideoProcessorTrack` used everywhere else, so GenieX inference
    works identically regardless of video source.
  - New routes: `GET /depth`, `GET /depth.css`, `GET /depth.js`,
    `GET /ws/orbbec`.
  - `on_shutdown()` now also releases the Orbbec camera.
- **`pyproject.toml`**: added an optional dependency group,
  `depth-camera = ["pyorbbecsdk2>=2.1.2"]` — *not* a hard dependency, so
  `pip install -e .` for the plain webcam/RTSP demo is unaffected on
  hardware without an Orbbec camera. Install with `pip install -e ".[depth-camera]"`.
- **`.gitignore`**: added `Log/` — the Orbbec SDK writes its own runtime log
  there by default (CWD-relative), unrelated to this repo's own logging.

## 6. Prerequisites / one-time device setup (not tracked in git)

These live on the device, not in the repo:

1. **Orbbec SDK v2** (`.deb` install) + **OrbbecViewer** — see
   `github.com/orbbec/OrbbecSDK_v2` releases, `arm64.deb` for this device.
   Provides the udev rules the camera needs (`/etc/udev/rules.d/99-obsensor-libusb.rules`)
   and the GUI viewer (`/usr/local/bin/OrbbecViewer`) used to sanity-check
   the camera outside this webui.
2. **Firmware ≥ 1.4.53** (recommended 1.5.02) — Orbbec's documented minimum
   for the Gemini 2L on SDK v2. This device's camera shipped on 1.4.38 and
   was updated via OrbbecViewer's Firmware panel (Orbbec's official GUI-based
   upgrade flow — see `github.com/orbbec/OrbbecFirmware` for the file and
   the upgrade guide PDF).
3. **`pyorbbecsdk2`** in the venv: `pip install --no-deps pyorbbecsdk2`.
   Deliberately `--no-deps` — the PyPI package pulls in `pygame`, `open3d`,
   and `pynput` (→ `evdev`, which needs kernel headers to build and fails on
   this image) for its *bundled example scripts*; the actual importable
   module (`pyorbbecsdk`) only needs `numpy`/`opencv-python`/`av`, already
   satisfied by this project's own dependencies.
4. **Only one process can hold the camera open at a time.** If `OrbbecViewer`
   is running, `/ws/orbbec` / the Color WebRTC track will fail to open the
   device (`uvc_open failed`). Close the Viewer before starting the webui's
   Orbbec mode, or vice versa.

## 7. `/depth` page — what's there

Layout: 2×2 grid — **Color**, **Depth** (JET-colormapped), **IR**, and
**VLM Output** — plus a slide-out side drawer.

- **Color** — live WebRTC feed, same VLM pipeline as the main page (periodic
  frame capture → GenieX → answer).
- **Depth** — colorized depth map, pushed over `/ws/orbbec`.
- **IR** — the infrared feed; has a header toggle to swap that panel to a
  quick-glance **System** view (CPU/RAM ring gauges + CPU/iGPU/NPU thermal
  history chart) without leaving the grid.
- **VLM Output** — model name + latency/avg/count metrics in the header,
  current answer (with pop-in/glow animation and Answer Length-aware prompt
  echo), and a 2-answer scrollback history below it — same UX as the main
  page's result panel, ported over.
- **Fullscreen** — each of the three feed panels (Color/Depth/IR) has its own
  fullscreen toggle (CSS-only overlay, not the Fullscreen API — same
  approach the main page uses for its video card).
- **Side drawer**, two tabs:
  - **Local System Stats** — host identity (board/SoC/CPU/GPU/NPU model,
    hostname, OS, kernel — same fields the main page's system-stats card
    shows) plus the Orbbec camera's own telemetry (device name/serial/
    firmware/connection, accelerometer, gyroscope, IMU temperature).
  - **Prompt** — full prompt editor ported from the main page: preset
    dropdown, custom prompt textarea, Answer Length dropdown, Max Tokens.
    Same protocol (`update_prompt` over `/ws`), same behavior (structured
    presets like Robot Navigation/OCR disable the length dropdown since a
    sentence cap would break their output format).

## 8. Feature parity with the main page — what's intentionally *not* here

Ported deliberately; skipped items were judged lower-value for a
camera-test page and can be added if actually wanted:

| Main page feature | On `/depth`? |
|---|---|
| Answer scrollback (last 2 answers) | ✅ ported |
| Pop-in/glow animation on new answers | ✅ ported |
| Model name + latency/avg/count | ✅ ported |
| Prompt editor (presets/custom/length/tokens) | ✅ ported |
| CPU/RAM ring gauges + thermal chart | ✅ ported |
| Fullscreen video | ✅ ported (all 3 feeds, not just one) |
| Markdown rendering of answers | ❌ plain text only |
| Copy-to-clipboard button | ❌ not added |
| Debug request/response payload viewer | ❌ not added |
| Guided UI tour | ❌ not added |
| Settings modal (theme, layout options, etc.) | ❌ not added |

## 9. Known issues / things to check next time this branch is picked up

- **Camera firmware**: was below Orbbec's minimum when first connected
  (1.4.38 vs. 1.4.53 minimum / 1.5.02 recommended); updated during
  development. If a *different* Gemini 2L unit is used at Electronica,
  check its firmware version the same way (OrbbecViewer → More → Device
  Info → FW Version) before relying on it.
- **Color feed lag**: was traced to USB bandwidth/CPU contention with the
  main page's webcam tab running at the same time (both pulling from USB,
  plus this process's own JPEG/base64 encoding for Depth+IR). Resolved by
  not running both video sources simultaneously. If lag reappears, check
  what else is pulling from USB/CPU before assuming it's a regression.
- **GUI automation for firmware updates / browser testing was unreliable**
  in this dev environment (Wayland-native Firefox, portal file-choosers) —
  not a product issue, just a note that manual verification in an actual
  browser is the more reliable check than scripted automation here.
