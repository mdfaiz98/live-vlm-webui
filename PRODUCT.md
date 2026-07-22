# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary audience: prospective customers and partners watching a live trade-show/sales demo booth pitch, with an Advantech presenter driving the UI. The interface must read clearly and impressively at a glance on a booth display, often viewed from a short distance by people who are not engineers. Secondary audience (not primary for this redesign pass): internal/customer engineers doing hands-on technical evaluation, who need the denser data (latency, thermal, prompt/response detail) still present and legible.

## Product Purpose

A fully offline, on-device Vision-Language-Model demo proving real-time VLM inference runs on Advantech's own edge NPU hardware (AIR-055 / Qualcomm QCS9075 Dragonwing) with no cloud dependency. Success is a booth visitor immediately understanding "this AI is running live, on this box, with no internet" and being impressed enough to engage further.

## Positioning

The differentiator is edge-native, fully offline real-time VLM inference on Advantech's own NPU silicon — a capability neighboring booths running cloud-hosted or GPU-server demos cannot truthfully claim. The UI should visibly reinforce "offline," "real-time," and "on-device" (e.g. live hardware telemetry, live camera-to-response loop) rather than reading as a generic chat/dashboard UI.

## Operating Context

Runs as three manually-started local processes (`geniex serve`, `geniex_proxy.py`, the web server) on the AIR-055 device itself, viewed in a browser on a connected display at a trade-show booth. WebRTC video from a webcam, RTSP camera, or local video file is streamed to the on-device Qwen3-VL-4B model; the model's text response streams back with latency metrics. A presenter operates the controls live in front of visitors, so the primary flow (start stream → see VLM output) must be operable with minimal fumbling mid-pitch.

## Capabilities and Constraints

- Video sources: webcam, RTSP IP camera (beta), local video file upload.
- Prompt editor: quick presets (scene description, object detection, activity/emotion recognition, safety monitoring, OCR, yes/no Q&A, robot navigation) plus custom prompt and max-tokens control.
- VLM output panel: live response text (markdown-capable), current prompt, latency/avg-latency/count metrics, optional request/response JSON debug payloads.
- Local system stats: CPU utilization, system RAM, and thermal monitoring split by CPU / iGPU / NPU (NPU thermal is a meaningful signal — it visibly proves the model is running on-device hardware, not the cloud).
- Settings modal: layout order, VLM-output-on-video overlay, motion/glow/fade toggles, WebRTC max latency, stats refresh interval, debug payload visibility.
- Constraint: no internet dependency anywhere in the running demo (no CDN assets, no STUN/TURN, no cloud calls) — this is both a technical and a positioning constraint.
- Occasional GenieX inference stalls (2-6+ min) are a known upstream quirk mitigated by a timeout+cooldown; the UI should degrade gracefully (visible status, not a frozen/broken-looking state) during this.

## Brand Commitments

Advantech branding: Advantech logo in the header (`static/images/advantech-logo.png` / new `advantech-logo1.jpg`), product title "Live VLM". Existing dark theme uses a green accent (`#35E28A`) and a technical/industrial visual language (sharp low-radius corners, monospace accents) established in the `new-design` branch redesign. This is evidence of current direction, not a locked constraint for further exploration.

## Evidence on Hand

Live, working product (not a mockup) at `src/live_vlm_webui/static/` — real functioning video pipeline, real system telemetry, real model output. No fabricated metrics/testimonials should be introduced; any numbers shown in mockups should be plausible placeholders clearly serving as UI demonstration, not claimed real benchmarks.

## Product Principles

1. Prove "live, offline, on-device" visually and immediately — hardware telemetry (especially NPU) is a feature, not an afterthought.
2. Optimize the primary loop (start → camera → VLM response) for a presenter driving the pitch, not a self-serve technical user.
3. Keep dense technical detail (metrics, debug payloads, settings) available but secondary — booth legibility from a few feet away takes priority over information density.
4. Never reintroduce an internet dependency (fonts, icons, STUN/TURN, cloud calls) — the offline story is part of the product truth.
5. Preserve Advantech brand presence (logo, name) as a fixed anchor even if the rest of the visual world changes.

## Accessibility & Inclusion

No specific requirement established beyond general legibility at a distance on a booth display.
