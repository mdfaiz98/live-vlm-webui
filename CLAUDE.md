# Advantech Live VLM Demo — Project Context

Fork of NVIDIA-AI-IOT/live-vlm-webui, rebranded and hardened for a fully
offline, on-device VLM demo on an Advantech AIR-055 (Qualcomm QCS9075 /
IQ-9 Dragonwing, 100 TOPS variant). Model: Qwen3-VL-4B-Instruct via GenieX,
running on the Hexagon NPU.

Branches: `main` = stable/working. `newgui` = Advantech-branded UI
(collapsible drawer, resizable panels, circular gauges, thermal monitoring).

## Architecture

```
Browser (WebRTC) <--> live-vlm-webui server.py <--> geniex_proxy.py (18182) <--> geniex serve (18181, NPU)
```

`geniex_proxy.py` (in `scripts/`) is required — GenieX's `/v1/models` advertises
a precision-tagged model name (`qualcomm/Qwen3-VL-4B-Instruct:W4A16`) that
`/v1/chat/completions` then rejects. The proxy strips the tag before
forwarding. Always point clients at port 18182, never 18181 directly.

## Known GenieX quirks (not our bugs — upstream, developer preview software)

- Model-tag mismatch above — worked around by the proxy.
- `aioice` (the WebRTC library GenieX/aiortc uses) hard-excludes the loopback
  interface from candidate gathering, with no config to override (open GitHub
  issue since 2018). Do NOT attempt to force loopback-only WebRTC — it will
  leave zero usable candidates and break the connection completely. This was
  tried and reverted.
- Occasionally a single `/v1/chat/completions` call takes 2-6+ minutes
  instead of the normal 2-4s, for reasons not yet root-caused (thermal
  throttling, NPU resource contention, and fastRPC allocation hiccups are all
  plausible; unconfirmed). `vlm_service.py` now has a 25s client timeout PLUS
  a 40s cooldown after any failure - the cooldown is essential: a timeout
  without it causes request pile-up (GenieX keeps processing abandoned
  requests, each new frame queues up behind them, causing accelerating delays
  and `broken pipe` errors). Don't remove the cooldown without addressing the
  pile-up risk.
- Gemma GGUF models (google/gemma-4-E4B-it-qat-q4_0-gguf) fail to load on the
  Hexagon NPU backend (`ggml-hexagon`) with a `fastrpc_mmap failed` buffer
  allocation error, but work fine on `--compute cpu`. Likely due to the
  model's non-uniform QAT quantization scheme; llama.cpp/Hexagon NPU support
  is newer/less mature than the QAIRT path Qwen uses.
- Qwen3-VL-8B-Instruct is listed on Qualcomm AI Hub's website with IQ-9075
  support, but `geniex pull ai-hub-models/Qwen3-VL-8B-Instruct` returns
  "not found" - the website catalog and GenieX's actual pull-able catalog are
  out of sync at this stage. Not fixable from our side; recheck periodically.

## Frontend fixes already applied (don't reintroduce these)

- Hardcoded Google STUN servers removed from both webcam and RTSP WebRTC
  configs (browser-side `index.html`, server-side `server.py`) - this is a
  same-machine connection, STUN/TURN are never needed and silently broke
  offline operation.
- Three CDN-loaded libraries (lucide, marked, dompurify) are now self-hosted
  in `static/vendor/` and served via a `/vendor` static route - previously
  loaded from unpkg.com/cdn.jsdelivr.net, which stalled page init with no
  internet (render-blocking script tags with no async/defer).
- `server.py`'s `index()` handler injects the actual `--api-base` CLI arg into
  the served HTML, replacing a hardcoded Ollama-port default.
- Auto-reconnect: `attemptAutoReconnect()` in `index.html` retries every 3s on
  ICE disconnect/failed, distinguishing user-initiated stops (`userStop()`)
  from unexpected drops.

## Things NOT to do

- Don't add STUN/TURN servers back.
- Don't try to force WebRTC onto loopback (127.0.0.1) - aioice can't gather
  loopback candidates, confirmed dead end.
- Don't remove the vlm_service.py cooldown logic without replacing it with
  something that prevents request pile-up after a timeout.
- Don't rewrite prompts/max_tokens back to unbounded - Object Detection and
  Accessibility presets were deliberately capped to reduce generation time.
- Avoid systemd services for the demo itself (geniex serve / proxy / webui) -
  deliberately kept as plain manual commands so the whole demo transfers
  cleanly to different hardware with minimal system-level setup.

## Running the demo

See `demo_documentation.md` in the repo root for the full step-by-step
(three terminals: `geniex serve`, `scripts/geniex_proxy.py`,
`./scripts/start_server.sh --api-base http://127.0.0.1:18182/v1 --model qualcomm/Qwen3-VL-4B-Instruct`).
