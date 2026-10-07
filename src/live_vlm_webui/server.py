# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebRTC Live VLM WebUI Server
Main server that handles WebRTC connections and serves the web interface
"""

import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaRelay
from PIL import Image

from .vlm_service import VLMService
from .video_processor import VideoProcessorTrack
from .gpu_monitor import create_monitor
from .rtsp_track import RTSPVideoTrack
from .video_file_track import VideoFileTrack
from .detector import YoloDetector
from .detection_processor import DetectionVideoTrack

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global objects
relay = MediaRelay()
pcs = set()
vlm_service = None  # Kept for backwards compat; default session uses sessions["default"]
websockets = set()  # Track active WebSocket connections (all)
gpu_monitor = None  # GPU monitoring instance
gpu_monitor_task = None  # Background task for GPU monitoring
rtsp_tracks = {}  # Track active RTSP streams {session_id: (rtsp_track, processor_track)}

# Detection cascade demo (/traffic) - deliberately separate from the normal
# demo's session/VLM machinery above; see CLAUDE.md and
# VLM_Cascade_Architecture_Plan.md for why this is a standalone page.
_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAFFIC_MODEL_PATH = _REPO_ROOT / "models" / "yolo26n_det_qcs9075.tflite"
TRAFFIC_LABELS_PATH = _REPO_ROOT / "models" / "coco_labels.txt"
TRAFFIC_VIDEO_PATH = _REPO_ROOT / "demo_assets" / "car-highway.mp4"
TRAFFIC_MODELS_DIR = _REPO_ROOT / "models"
TRAFFIC_VIDEOS_DIR = _REPO_ROOT / "demo_assets"
TRAFFIC_UPLOADS_DIR = TRAFFIC_VIDEOS_DIR / "uploads"
# Entry zone per demo video (x1, y1, x2, y2 as fractions of the frame) - only
# vehicles inside it are drawn and logged (see DetectionVideoTrack.entry_zone).
# The parking clip's zone is the ground in front of the barrier: street traffic
# behind it has box bottoms at y < 0.27, cars at/through the barrier at
# y >= 0.3, and the car parked at the right edge sits at x > 0.9 (measured from
# every detection in the clip). Videos not listed use the whole frame.
TRAFFIC_ENTRY_ZONES = {
    "parking-entrance-barrier.mp4": (0.15, 0.28, 0.80, 1.0),
}
TRAFFIC_VIDEO_NAMES = {
    "car-highway.mp4": "Highway traffic",
    "parking-entrance-barrier.mp4": "Parking entrance (barrier)",
}
yolo_detector = None  # The active detector - loaded at startup (HTP delegate warmup is ~1-2s) - see on_startup
# Settings drawer: every .tflite in models/ is selectable as the detection
# model (all share coco_labels.txt and the YOLO26 export's boxes/scores/
# class_idx outputs). Loaded on first use, then kept so switching back is instant.
traffic_detectors = {}  # model file name -> YoloDetector
traffic_detection_model_id = TRAFFIC_MODEL_PATH.name
traffic_websockets = set()  # Track active WebSocket connections for the /traffic page
# The single-worker detection executor (see detection_processor.py) means
# multiple simultaneous /traffic connections serialize against each other
# for NPU access - if a reconnect creates a new connection before the old
# one is torn down, both compete for that one worker, frame delivery stalls,
# and WebRTC's own timeout closes the connection, triggering another
# reconnect (observed as a rapid connect/disconnect loop that eventually
# crashed the process). This is also just correct behavior for a
# single-screen kiosk demo: only one active /traffic connection at a time.
traffic_active_pc = None
traffic_active_video_track = None
traffic_active_processor_track = None  # the DetectionVideoTrack - see "toggle_detection"
# Opening/closing PyAV video containers back-to-back with no gap was
# observed to trigger a native heap-corruption crash ("munmap_chunk():
# invalid pointer") - this enforces a minimum spacing between /traffic
# connection setups server-side, independent of client behavior (a client
# retrying aggressively, e.g. a stale browser tab still running old JS
# without backoff, can't force the server back into that failure mode).
traffic_last_offer_time = 0.0
TRAFFIC_MIN_OFFER_INTERVAL_SECONDS = 2.0

# Phase 3: click-to-caption. traffic_vlm_service is a dedicated VLMService
# instance (separate cooldown/lock state from the main demo's, but pointed
# at the same GenieX server underneath) used only for per-vehicle captions,
# never for continuous full-frame analysis. traffic_track_store retains each
# reported vehicle's full-resolution crop (NOT the shrunk display thumbnail
# broadcast to the panel) keyed by track id, since a click can arrive well
# after tracker.py's own Track object has expired/been GC'd; capped and
# FIFO-evicted so a long-running demo can't grow this unboundedly.
traffic_vlm_service: Optional[VLMService] = None
traffic_track_store: "OrderedDict[str, dict]" = OrderedDict()
TRAFFIC_TRACK_STORE_MAX = 200
# Plate on its own line, with an explicit "not readable" answer: with only
# "don't guess" Qwen3-VL invented a plate for a car whose plate was behind the
# parking barrier arm; with this wording it says "not readable" there, and
# reads the plate (and brand) once the car is clear.
TRAFFIC_CAPTION_DEFAULT_PROMPT = (
    "Describe this vehicle in one sentence: color, type (car/SUV/van/truck/etc.) and brand "
    "if visible. Then on a new line write 'Plate: ' followed by the license plate exactly as "
    "shown, only if every character is clearly readable. If the plate is blocked, blurry, "
    "cut off or partly hidden, write 'Plate: not readable'. Never guess."
)
# The active prompt - editable at runtime from the /traffic page (see
# "update_prompt" in traffic_websocket_handler); TRAFFIC_CAPTION_DEFAULT_PROMPT
# above stays fixed as the "Reset to default" target.
traffic_caption_prompt = TRAFFIC_CAPTION_DEFAULT_PROMPT
TRAFFIC_CAPTION_MAX_TOKENS = 120
# Mirrors vlm_service.py's own cooldown - see CLAUDE.md: without this, a
# GenieX stall causes request pile-up (each new caption request queues up
# behind an abandoned one, accelerating delays and "broken pipe" errors).
TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS = 40.0
traffic_caption_last_failure_at: Optional[float] = None
traffic_caption_lock = asyncio.Lock()

# Multi-session state (0.4.0)
default_vlm_config = {}  # Set at startup; used to create new sessions
sessions = {}  # session_id -> {"vlm_service": VLMService}
session_websockets = defaultdict(set)  # session_id -> set of ws
ws_to_session = {}  # ws -> session_id


def get_or_create_session(session_id: str):
    """Get or create per-session state (VLM service). Thread-safe for aiohttp."""
    if session_id not in sessions:
        cfg = default_vlm_config
        sessions[session_id] = {
            "vlm_service": VLMService(
                model=cfg.get("model", "meta/llama-3.2-11b-vision-instruct"),
                api_base=cfg.get("api_base", "http://localhost:8000/v1"),
                api_key=cfg.get("api_key", "EMPTY"),
                prompt=cfg.get("prompt", "Describe what you see in this image in one sentence."),
            ),
            "show_request_payload": False,
            "show_response_payload": False,
        }
        logger.info(f"Created new session: {session_id}")
    return sessions[session_id]


def send_to_session(session_id: str, message: str):
    """Send a message only to WebSocket clients in this session."""
    for ws in session_websockets.get(session_id, set()):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")


def get_session_callback(session_id: str):
    """Return a text_callback that sends VLM results only to this session."""

    def callback(text: str, metrics: dict):
        out = {"type": "vlm_response", "text": text, "metrics": metrics}
        session = sessions.get(session_id)
        if session and session.get("vlm_service"):
            svc = session["vlm_service"]
            if session.get("show_request_payload"):
                payload = svc.get_last_request_payload()
                if payload is not None:
                    out["request_payload"] = payload
            if session.get("show_response_payload"):
                payload = svc.get_last_response_payload()
                if payload is not None:
                    try:
                        out["response_payload"] = json.loads(json.dumps(payload, default=str))
                    except (TypeError, ValueError):
                        out["response_payload"] = payload
        send_to_session(session_id, json.dumps(out))

    return callback


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available for binding"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def find_process_using_port(port):
    """Find what process is using a port (Linux/Unix only)"""
    try:
        # Try lsof first (more reliable)
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            # Get process name
            name_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True, timeout=2
            )
            if name_result.returncode == 0:
                return f"PID {pid} ({name_result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available, try netstat
        try:
            result = subprocess.run(
                ["netstat", "-tulpn"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        return parts[-1]  # PID/Program name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return "unknown process"


def find_available_port(start_port=8080, max_attempts=10):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


async def detect_local_service_and_model():
    """
    Auto-detect available local VLM services and select a model
    Returns: (api_base, model_name) or (None, None) if no service found
    """
    services = [
        ("http://127.0.0.1:18182/v1", "GenieX (via fix-up proxy)"),
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:30000/v1", "SGLang"),
    ]

    for api_base, service_name in services:
        try:
            # Try to connect to the service
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{api_base}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Prefer vision models
                            vision_keywords = ["vision", "llava", "llama-3.2", "gemini"]
                            for model in models:
                                model_id = model.get("id", "")
                                if any(keyword in model_id.lower() for keyword in vision_keywords):
                                    logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                                    logger.info(f"   Selected model: {model_id}")
                                    return (api_base, model_id)

                            # If no vision model found, use the first one
                            model_id = models[0].get("id", "")
                            if service_name.startswith("GenieX"):
                                # GenieX's own /v1/models advertises a precision-tagged
                                # id (e.g. ":W4A16") that /v1/chat/completions then
                                # rejects - strip it so auto-detect doesn't walk
                                # straight into that known bug.
                                model_id = model_id.split(":")[0]
                            logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                            logger.info(
                                f"   Selected model: {model_id} (vision model preferred but not found)"
                            )
                            return (api_base, model_id)
        except Exception as e:
            logger.debug(f"Service {service_name} not available at {api_base}: {e}")
            continue

    return (None, None)


async def index(request):
    """Serve the main HTML page, with the actual configured API base URL and
    model (from --api-base / --model at startup) injected in place of the
    hardcoded Ollama default - so the field is correct on page load without
    relying on auto-detection or manual entry."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r").read()

    configured_api_base = default_vlm_config.get("api_base")
    if configured_api_base:
        content = content.replace(
            'id="apiBaseUrl" value="http://localhost:11434/v1"',
            f'id="apiBaseUrl" value="{configured_api_base}"',
        )

    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def app_style(request):
    """Serve the extracted stylesheet (was previously inlined in index.html).

    Read fresh + no-store on every request, same as index() - this is an
    actively-edited demo, not a CDN asset, and browser caching here has
    previously made in-progress style/script edits look like they hadn't
    taken effect after a normal refresh."""
    path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    content = open(path, "r").read()
    return web.Response(content_type="text/css", text=content, headers={"Cache-Control": "no-store"})


async def app_script(request):
    """Serve the extracted app JS (was previously inlined in index.html). See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "app.js")
    content = open(path, "r").read()
    return web.Response(content_type="text/javascript", text=content, headers={"Cache-Control": "no-store"})


async def traffic_index(request):
    """Serve the /traffic cascade demo page - separate from index.html."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "traffic.html"), "r").read()
    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def traffic_style(request):
    """Serve the /traffic demo's stylesheet. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "traffic.css")
    content = open(path, "r").read()
    return web.Response(content_type="text/css", text=content, headers={"Cache-Control": "no-store"})


async def traffic_script(request):
    """Serve the /traffic demo's JS. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "traffic.js")
    content = open(path, "r").read()
    return web.Response(content_type="text/javascript", text=content, headers={"Cache-Control": "no-store"})


def broadcast_traffic_detections(tracks, crops, crops_large, frame_width, frame_height):
    """Send newly-confirmed vehicle sightings (with cropped thumbnails) to
    all connected /traffic WebSocket clients - called once per vehicle, the
    first time the tracker confirms it (see tracker.py), not once per frame.

    Two display sizes are sent: a small "crop" for the panel list, and a
    larger "crop_large" for the click-to-open detail modal - both are still
    below the full-resolution crop the VLM itself analyzes (retained
    separately in traffic_track_store), which is why the VLM can sometimes
    make out detail (e.g. plate text) a viewer can't see in the modal."""
    objects = [
        {
            "id": t.id,
            "label": t.label,
            "score": round(t.best_score, 3),
            "bbox": list(t.best_bbox),
            "crop": crop_b64,
            "crop_large": crop_large_b64,
        }
        for t, crop_b64, crop_large_b64 in zip(tracks, crops, crops_large)
    ]

    for t in tracks:
        traffic_track_store[t.id] = {
            "label": t.label,
            "score": t.best_score,
            "best_crop": t.best_crop,  # full-res ndarray, not the shrunk thumbnail
            # The tracker keeps refining t.best_crop after the card appears (e.g. a
            # car that waited behind the barrier arm, then drove through closer) -
            # Describe uses that latest crop, see caption_traffic_vehicle
            "track": t,
            "caption": None,
            "caption_pending": False,
        }
        traffic_track_store.move_to_end(t.id)
    while len(traffic_track_store) > TRAFFIC_TRACK_STORE_MAX:
        traffic_track_store.popitem(last=False)

    message = json.dumps({"type": "new_detections", "objects": objects})
    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error broadcasting traffic detection: {e}")


def broadcast_traffic_caption_status(track_id, status, caption=None):
    """Send a caption_pending/caption_result/caption_error update for one
    vehicle to all connected /traffic WebSocket clients."""
    message = json.dumps({"type": "caption_status", "id": track_id, "status": status, "caption": caption})
    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error broadcasting traffic caption status: {e}")


async def caption_traffic_vehicle(track_id: str):
    """Generate a VLM caption for one previously-detected vehicle's stored
    crop, on demand (click-to-caption). Serialized via traffic_caption_lock
    (one caption call at a time) and respects the same failure-cooldown
    pattern as vlm_service.py, to protect against GenieX request pile-up
    after a stall (see CLAUDE.md)."""
    global traffic_caption_last_failure_at

    entry = traffic_track_store.get(track_id)
    if entry is None:
        logger.debug(f"[traffic] Caption requested for unknown/expired track {track_id}")
        return
    if entry["caption"] is not None or entry["caption_pending"]:
        return  # already captioned or already in flight
    if traffic_vlm_service is None:
        broadcast_traffic_caption_status(track_id, "error", "Captioning unavailable")
        return

    if traffic_caption_last_failure_at is not None:
        elapsed = time.time() - traffic_caption_last_failure_at
        if elapsed < TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS:
            logger.info(
                f"[traffic] In cooldown after a recent VLM failure "
                f"({elapsed:.0f}s / {TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS:.0f}s), "
                f"skipping caption request for {track_id}"
            )
            broadcast_traffic_caption_status(track_id, "error", "Busy, try again shortly")
            return

    if traffic_caption_lock.locked():
        broadcast_traffic_caption_status(track_id, "error", "Busy, try again shortly")
        return

    entry["caption_pending"] = True
    broadcast_traffic_caption_status(track_id, "pending")

    async with traffic_caption_lock:
        try:
            crop = entry["track"].best_crop if entry["track"].best_crop is not None else entry["best_crop"]
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            caption = await traffic_vlm_service.analyze_image(image, prompt=traffic_caption_prompt)
            if caption.startswith("Error:"):
                raise RuntimeError(caption)
            entry["caption"] = caption
            entry["caption_pending"] = False
            broadcast_traffic_caption_status(track_id, "done", caption)
        except Exception as e:
            logger.error(f"[traffic] Caption failed for {track_id}: {e}")
            traffic_caption_last_failure_at = time.time()
            entry["caption_pending"] = False
            broadcast_traffic_caption_status(track_id, "error", "Description failed")


def traffic_list_detection_models():
    """Selectable detection models: every .tflite in models/, e.g.
    yolo26n_det_qcs9075.tflite -> "YOLO26n"."""
    models = []
    for path in sorted(TRAFFIC_MODELS_DIR.glob("*.tflite")):
        m = re.match(r"(yolo)(v?\d+)([nsmlx])", path.stem, re.IGNORECASE)
        name = f"YOLO{m.group(2)}{m.group(3).lower()}" if m else path.stem
        models.append({"id": path.name, "name": name})
    return models


def traffic_get_detector(model_id: str) -> YoloDetector:
    """Load (once) and return the detector for a models/ file. Blocking - call
    via run_in_executor."""
    if model_id not in traffic_detectors:
        traffic_detectors[model_id] = YoloDetector(
            model_path=str(TRAFFIC_MODELS_DIR / model_id),
            labels_path=str(TRAFFIC_LABELS_PATH),
            backend="htp",
        )
    return traffic_detectors[model_id]


def _video_files(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]


def traffic_list_videos():
    """Videos selectable as the /traffic source: the demo videos in
    demo_assets/, then videos uploaded from the settings drawer
    (demo_assets/uploads/ - kept apart from the main demo's upload folder)."""
    videos = [
        {"id": str(p), "name": TRAFFIC_VIDEO_NAMES.get(p.name, p.stem.replace("-", " ").capitalize())}
        for p in _video_files(TRAFFIC_VIDEOS_DIR)
    ]
    videos += [{"id": str(p), "name": f"{p.name} (uploaded)"} for p in _video_files(TRAFFIC_UPLOADS_DIR)]
    return videos


async def traffic_upload(request):
    """POST /api/traffic/upload (multipart field "file") - add a video to the
    /traffic list, saved under its own (sanitized) name in demo_assets/uploads/."""
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"error": "Expected a multipart field named 'file'"}, status=400)
    original = field.filename or "video.mp4"
    stem, ext = os.path.splitext(os.path.basename(original))
    if ext.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return web.json_response(
            {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}, status=400
        )
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:60] or "video"
    TRAFFIC_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path, n = TRAFFIC_UPLOADS_DIR / f"{stem}{ext.lower()}", 2
    while path.exists():  # never overwrite an earlier upload
        path, n = TRAFFIC_UPLOADS_DIR / f"{stem}-{n}{ext.lower()}", n + 1
    size = 0
    with open(path, "wb") as f:
        while chunk := await field.read_chunk(size=1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_UPLOAD_BYTES:
                f.close()
                path.unlink()
                return web.json_response({"error": "File too large (max 2GB)"}, status=413)
            f.write(chunk)
    logger.info(f"[traffic] Video uploaded: {path} ({size} bytes)")
    return web.json_response({"id": str(path), "name": f"{path.name} (uploaded)"})


async def traffic_settings(request):
    """GET /api/traffic/settings - options + current values for the settings drawer."""
    vlm_models = []
    if traffic_vlm_service is not None:
        try:
            listed = await traffic_vlm_service.client.models.list()
            # GenieX advertises a precision tag ("...:W4A16") the proxy strips anyway
            vlm_models = sorted({m.id.split(":")[0] for m in listed.data})
        except Exception as e:
            logger.warning(f"[traffic] Could not list VLM models: {e}")
        if traffic_vlm_service.model not in vlm_models:
            vlm_models.insert(0, traffic_vlm_service.model)
    return web.json_response(
        {
            "detection_models": traffic_list_detection_models(),
            "detection_model": traffic_detection_model_id,
            "vlm_models": [{"id": m, "name": m.split("/")[-1]} for m in vlm_models],
            "vlm_model": traffic_vlm_service.model if traffic_vlm_service else None,
            "videos": traffic_list_videos(),
            "default_video": str(TRAFFIC_VIDEO_PATH),
        }
    )


async def traffic_set_detection_model(model_id: str):
    """Switch the active detection model (settings drawer), live if streaming."""
    global yolo_detector, traffic_detection_model_id
    if model_id not in {m["id"] for m in traffic_list_detection_models()}:
        return
    try:
        detector = await asyncio.get_event_loop().run_in_executor(None, traffic_get_detector, model_id)
    except Exception as e:
        logger.error(f"[traffic] Failed to load detection model {model_id}: {e}")
        message = {"type": "settings_error", "text": f"Could not load {model_id}"}
    else:
        yolo_detector, traffic_detection_model_id = detector, model_id
        if traffic_active_processor_track is not None:
            traffic_active_processor_track.detector = detector
        logger.info(f"[traffic] Detection model set to {model_id}")
        message = {"type": "settings_updated", "detection_model": model_id}
    for ws in list(traffic_websockets):
        asyncio.create_task(ws.send_json(message))


async def traffic_websocket_handler(request):
    """WebSocket for the /traffic demo - detection results, plus pause/resume
    control of the video source (see VideoFileTrack.pause/resume) and
    editing the click-to-caption prompt. No per-session prompt/model
    machinery like the main demo's /ws; that's the normal demo's concern."""
    global traffic_caption_prompt
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    traffic_websockets.add(ws)
    logger.info(f"Traffic WebSocket client connected, total: {len(traffic_websockets)}")

    try:
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to traffic demo",
                "prompt": traffic_caption_prompt,
                "default_prompt": TRAFFIC_CAPTION_DEFAULT_PROMPT,
            }
        )
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type")
                if msg_type in ("pause", "resume") and traffic_active_video_track:
                    getattr(traffic_active_video_track, msg_type)()
                elif msg_type == "toggle_detection" and traffic_active_processor_track:
                    if data.get("enabled"):
                        traffic_active_processor_track.enable_detection()
                    else:
                        traffic_active_processor_track.disable_detection()
                elif msg_type == "request_caption":
                    track_id = data.get("id")
                    if track_id:
                        asyncio.create_task(caption_traffic_vehicle(track_id))
                elif msg_type == "set_detection_model":
                    asyncio.create_task(traffic_set_detection_model(data.get("id", "")))
                elif msg_type == "set_vlm_model" and traffic_vlm_service is not None:
                    model_id = (data.get("id") or "").strip()
                    if model_id:
                        traffic_vlm_service.model = model_id
                        logger.info(f"[traffic] VLM model set to {model_id}")
                        for other_ws in list(traffic_websockets):
                            asyncio.create_task(
                                other_ws.send_json({"type": "settings_updated", "vlm_model": model_id})
                            )
                elif msg_type == "update_prompt":
                    new_prompt = (data.get("prompt") or "").strip()
                    if new_prompt:
                        traffic_caption_prompt = new_prompt
                        logger.info(f"[traffic] Caption prompt updated: {new_prompt!r}")
                        for other_ws in list(traffic_websockets):
                            try:
                                asyncio.create_task(
                                    other_ws.send_json(
                                        {"type": "prompt_updated", "prompt": traffic_caption_prompt}
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error broadcasting prompt update: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"Traffic WebSocket error: {ws.exception()}")
    finally:
        traffic_websockets.discard(ws)
        logger.info(f"Traffic WebSocket client disconnected, total: {len(traffic_websockets)}")

    return ws


async def traffic_offer(request):
    """Handle WebRTC offer for the /traffic cascade demo, wrapping the chosen
    source video with object detection. The video (settings drawer) is any
    file from traffic_list_videos(), looped (default: car-highway.mp4).
    Kept deliberately separate from the normal demo's offer()/VLM flow (see
    CLAUDE.md)."""
    if yolo_detector is None:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {"error": "Detection model not available - check server startup logs"}
            ),
        )

    global traffic_active_pc, traffic_active_video_track, traffic_active_processor_track, traffic_last_offer_time

    # Rate-limit connection setup itself (see TRAFFIC_MIN_OFFER_INTERVAL_SECONDS
    # comment above) - protects the server even if a client hammers this
    # endpoint (e.g. a stale tab stuck retrying every few seconds).
    now = time.monotonic()
    elapsed = now - traffic_last_offer_time
    if elapsed < TRAFFIC_MIN_OFFER_INTERVAL_SECONDS:
        await asyncio.sleep(TRAFFIC_MIN_OFFER_INTERVAL_SECONDS - elapsed)
    traffic_last_offer_time = time.monotonic()

    # Close any existing /traffic connection first (see traffic_active_pc
    # comment above) - a page reload should cleanly supersede the old
    # connection, never run alongside it.
    #
    # NOTE: closing the old pc/video track synchronously here, immediately
    # before opening a new VideoFileTrack on the same file, was observed to
    # trigger a native heap-corruption crash ("munmap_chunk(): invalid
    # pointer") under rapid reconnect churn - almost certainly a PyAV/libav
    # thread-safety issue with tearing down and standing up decode contexts
    # for the same file back-to-back with no gap. Stopping the old track and
    # yielding back to the event loop before continuing gives its native
    # cleanup a chance to finish first.
    if traffic_active_pc is not None:
        logger.info("[traffic] New connection superseding existing one - closing old first")
        old_pc, old_track = traffic_active_pc, traffic_active_video_track
        traffic_active_pc, traffic_active_video_track, traffic_active_processor_track = None, None, None
        if old_track:
            old_track.stop()
        pcs.discard(old_pc)
        await old_pc.close()
        await asyncio.sleep(0.2)

    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    video_path = params.get("video") or str(TRAFFIC_VIDEO_PATH)
    if video_path not in {v["id"] for v in traffic_list_videos()}:
        return web.json_response({"error": "Unknown video file"}, status=400)

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    traffic_active_pc = pc

    video_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        global traffic_active_pc, traffic_active_video_track, traffic_active_processor_track
        logger.info(f"[traffic] Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            if video_cleanup_track:
                video_cleanup_track.stop()
            await pc.close()
            pcs.discard(pc)
            if traffic_active_pc is pc:
                traffic_active_pc = None
                traffic_active_video_track = None
                traffic_active_processor_track = None

    try:
        video_track = VideoFileTrack(video_path, loop=True)
        video_cleanup_track = video_track
        traffic_active_video_track = video_track
    except Exception as e:
        logger.error(f"[traffic] Failed to open demo video: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": f"Failed to open demo video: {str(e)}"}),
        )

    # buffered=False: keep only the latest frame, don't queue every frame the
    # relay pulls. DetectionVideoTrack does real per-frame work (NPU
    # inference + drawing + re-encoding) that can run slightly behind 30fps,
    # and the default buffered=True relay queue is unbounded - a backlog of
    # stale frames would build up over time, and pause() would then have to
    # drain that whole backlog (showing "old" motion) before actually
    # freezing. Dropping stale frames instead is also just correct for a
    # real-time low-latency pipeline like this one.
    relayed_video = relay.subscribe(video_track, buffered=False)
    processor_track = DetectionVideoTrack(
        relayed_video, yolo_detector, detection_callback=broadcast_traffic_detections
    )
    processor_track.set_entry_zone(TRAFFIC_ENTRY_ZONES.get(Path(video_path).name))
    if processor_track.entry_zone is not None:
        # On a loop, the last car of one pass and the first car of the next sit
        # in the same spot at the barrier - the tracker would take them for one
        # car (already logged) and skip the new one. Videos without a zone (the
        # highway) keep their original behaviour.
        video_track.on_loop = processor_track.reset_tracker
    traffic_active_processor_track = processor_track
    pc.addTrack(processor_track)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("[traffic] Created answer for traffic demo peer connection")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def models(request):
    """Return available models from the VLM API"""
    try:
        # Check if custom API base and key are provided in query params
        api_base = request.rel_url.query.get("api_base")
        api_key = request.rel_url.query.get("api_key")

        if api_base:
            # Query models from the provided API endpoint
            from openai import AsyncOpenAI

            temp_client = AsyncOpenAI(base_url=api_base, api_key=api_key if api_key else "EMPTY")
            models_response = await temp_client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": False}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
        else:
            # Use default session's VLM service (backwards compat when no api_base in query)
            default_svc = get_or_create_session("default")["vlm_service"]
            models_response = await default_svc.client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == default_svc.model}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return current model as fallback
        if sessions.get("default"):
            default_svc = sessions["default"]["vlm_service"]
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "models": [
                            {"id": default_svc.model, "name": default_svc.model, "current": True}
                        ]
                    }
                ),
            )
        return web.Response(
            content_type="application/json", text=json.dumps({"models": [], "error": str(e)})
        )


async def detect_services(request):
    """Detect available local VLM services"""
    services = [
        {
            "name": "GenieX (via fix-up proxy)",
            "url": "http://localhost:18182/v1",
            "port": 18182,
            "path": "/v1/models",
        },
        {"name": "Ollama", "url": "http://localhost:11434/v1", "port": 11434, "path": "/api/tags"},
        {"name": "vLLM", "url": "http://localhost:8000/v1", "port": 8000, "path": "/v1/models"},
        {"name": "SGLang", "url": "http://localhost:30000/v1", "port": 30000, "path": "/v1/models"},
    ]

    detected = []

    async def check_service(service):
        """Check if a service is running by probing its endpoint"""
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://localhost:{service['port']}{service['path']}"
                async with session.get(url) as response:
                    if response.status in [200, 404]:  # 404 is ok, means server is running
                        logger.info(f"Detected {service['name']} at {service['url']}")
                        return service
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        return None

    # Check all services concurrently
    results = await asyncio.gather(*[check_service(s) for s in services])
    detected = [s for s in results if s is not None]

    # Default to NVIDIA API Catalog if no local services found
    if not detected:
        detected.append(
            {
                "name": "NVIDIA API Catalog",
                "url": "https://integrate.api.nvidia.com/v1",
                "port": None,
                "path": None,
                "requires_key": True,
            }
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"detected": detected, "default": detected[0] if detected else None}),
    )


async def websocket_handler(request):
    """Handle WebSocket connections for text updates. Supports ?session_id= for multi-session."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session ID from query or generate new (client should send same id in /offer)
    session_id = request.query.get("session_id", "").strip() or str(uuid.uuid4())
    ws_to_session[ws] = session_id
    session_websockets[session_id].add(ws)
    websockets.add(ws)
    logger.info(
        f"WebSocket client connected. session_id={session_id}, total clients: {len(websockets)}"
    )

    session = get_or_create_session(session_id)
    svc = session["vlm_service"]

    try:
        # Send initial message with current server configuration (include session_id if we generated it)
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to server",
                "status": "Ready",
                "session_id": session_id,
            }
        )

        # Send current server configuration for this session
        from .video_processor import VideoProcessorTrack as _VPT

        await ws.send_json(
            {
                "type": "server_config",
                "model": svc.model,
                "api_base": svc.api_base,
                "prompt": svc.prompt,
                "process_every": _VPT.process_every_n_frames,
                "session_id": session_id,
            }
        )

        # Keep connection alive and handle incoming messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Re-resolve session in case it was recreated
                    svc = get_or_create_session(session_id)["vlm_service"]

                    if data.get("type") == "update_prompt":
                        new_prompt = data.get("prompt", "").strip()
                        max_tokens = data.get("max_tokens")
                        if new_prompt and svc:
                            svc.update_prompt(new_prompt, max_tokens)
                            logger.info(
                                f"[{session_id}] Prompt updated: {new_prompt}, max_tokens: {max_tokens}"
                            )

                            await ws.send_json(
                                {
                                    "type": "prompt_updated",
                                    "prompt": new_prompt,
                                    "max_tokens": max_tokens,
                                }
                            )

                    elif data.get("type") == "update_model":
                        new_model = data.get("model", "").strip()
                        api_base = data.get("api_base", "").strip()
                        api_key = data.get("api_key", "").strip()

                        if new_model and svc:
                            svc.model = new_model
                            if api_base:
                                svc.update_api_settings(api_base, api_key if api_key else None)
                                logger.info(
                                    f"[{session_id}] Model updated: {new_model}, API: {api_base}"
                                )
                            else:
                                logger.info(f"[{session_id}] Model updated: {new_model}")

                            await ws.send_json(
                                {
                                    "type": "model_updated",
                                    "model": new_model,
                                    "api_base": svc.api_base,
                                }
                            )

                    elif data.get("type") == "update_processing":
                        process_every = data.get("process_every", 30)
                        try:
                            process_every = int(process_every)
                            if 1 <= process_every <= 3600:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.process_every_n_frames
                                VideoProcessorTrack.process_every_n_frames = process_every
                                logger.info(
                                    f"[{session_id}] Processing interval updated: {old_value} → {process_every} frames"
                                )

                                await ws.send_json(
                                    {"type": "processing_updated", "process_every": process_every}
                                )
                            else:
                                logger.warning(
                                    f"Processing interval out of range (1-3600): {process_every}"
                                )
                        except ValueError:
                            logger.error(f"Invalid processing interval: {process_every}")

                    elif data.get("type") == "set_debug":
                        session_data = get_or_create_session(session_id)
                        if "show_request_payload" in data:
                            session_data["show_request_payload"] = bool(
                                data["show_request_payload"]
                            )
                        if "show_response_payload" in data:
                            session_data["show_response_payload"] = bool(
                                data["show_response_payload"]
                            )
                        logger.debug(
                            f"[{session_id}] Debug: request_payload="
                            f"{session_data.get('show_request_payload')}, response_payload="
                            f"{session_data.get('show_response_payload')}"
                        )

                    elif data.get("type") == "update_max_latency":
                        max_latency = data.get("max_latency", 0.0)
                        try:
                            max_latency = float(max_latency)
                            if 0 <= max_latency <= 10.0:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.max_frame_latency
                                VideoProcessorTrack.max_frame_latency = max_latency
                                status = "disabled" if max_latency == 0 else f"{max_latency:.1f}s"
                                old_status = "disabled" if old_value == 0 else f"{old_value:.1f}s"
                                logger.info(
                                    f"[{session_id}] Max frame latency updated: {old_status} → {status}"
                                )

                                await ws.send_json(
                                    {"type": "max_latency_updated", "max_latency": max_latency}
                                )
                            else:
                                logger.warning(f"Max latency out of range (0-10.0): {max_latency}")
                        except ValueError:
                            logger.error(f"Invalid max latency value: {max_latency}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        session_websockets[session_id].discard(ws)
        ws_to_session.pop(ws, None)
        websockets.discard(ws)
        logger.info(
            f"WebSocket client disconnected. session_id={session_id}, total clients: {len(websockets)}"
        )

    return ws


def broadcast_text_update(text: str, metrics: dict):
    """Broadcast text update and metrics to all connected WebSocket clients"""
    if not websockets:
        return

    message = json.dumps({"type": "vlm_response", "text": text, "metrics": metrics})

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            # Use asyncio to send without blocking
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


def broadcast_gpu_stats(stats: dict):
    """Broadcast GPU stats to all connected WebSocket clients (main demo and /traffic)"""
    if not websockets and not traffic_websockets:
        return

    message = json.dumps({"type": "gpu_stats", "stats": stats})

    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending GPU stats to traffic websocket: {e}")

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending GPU stats to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


async def gpu_monitor_loop():
    """Background task to periodically collect and broadcast GPU stats"""
    global gpu_monitor

    if not gpu_monitor:
        logger.warning("GPU monitor not initialized, skipping monitoring")
        return

    logger.info("GPU monitoring loop started")

    loop = asyncio.get_event_loop()
    try:
        while True:
            # Get current stats - off the event loop: reading /proc and sysfs
            # (~30 ms on the AFE-A503) mustn't delay video frames
            stats = await loop.run_in_executor(None, gpu_monitor.get_stats)

            # Update history with current stats
            gpu_monitor.update_history(stats)

            # Add history to stats
            stats["history"] = gpu_monitor.get_history()

            # Broadcast to all connected clients
            broadcast_gpu_stats(stats)

            # Update every 0.25 seconds for detailed GPU monitoring
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        logger.info("GPU monitoring loop cancelled")
    except Exception as e:
        logger.error(f"Error in GPU monitoring loop: {e}")


async def offer(request):
    """Handle WebRTC offer from client (supports both webcam and RTSP). Uses session_id for per-session VLM."""
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    rtsp_url = params.get("rtsp_url")  # Optional RTSP URL for IP camera mode
    video_file_path = params.get("video_file_path")  # Optional local video file mode
    session_id = params.get("session_id", "default")

    session = get_or_create_session(session_id)
    session_vlm = session["vlm_service"]
    session_callback = get_session_callback(session_id)

    # No STUN servers - this connection is always browser-to-same-machine (or
    # same-LAN), so host ICE candidates alone are sufficient. Public STUN
    # servers here would make the video feed depend on internet access for
    # no reason and break fully-offline operation.
    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    # Store RTSP track for cleanup
    rtsp_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            # Clean up RTSP track if exists
            if rtsp_cleanup_track:
                rtsp_cleanup_track.stop()
                logger.info("RTSP track stopped on connection close")
            await pc.close()
            pcs.discard(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.error("ICE connection failed - check firewall/NAT settings")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state: {pc.iceGatheringState}")

    # If RTSP URL provided, create RTSP track instead of waiting for browser track
    if rtsp_url:
        logger.info(f"[{session_id}] Creating RTSP track for: {rtsp_url}")
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
            rtsp_cleanup_track = rtsp_track  # Store for cleanup

            # Wait for initial connection to get stream info
            await asyncio.sleep(0.5)

            # Wrap RTSP track with relay first (same pattern as webcam)
            relayed_rtsp = relay.subscribe(rtsp_track)

            processor_track = VideoProcessorTrack(
                relayed_rtsp, session_vlm, text_callback=session_callback
            )

            # Add processor directly to peer connection
            pc.addTrack(processor_track)
            logger.info("Added RTSP processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )
    elif video_file_path:
        logger.info(f"[{session_id}] Creating video file track for: {video_file_path}")

        if not os.path.isfile(video_file_path):
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": f"Video file not found: {video_file_path}"}),
            )

        try:
            video_track = VideoFileTrack(video_file_path, loop=True)
            rtsp_cleanup_track = video_track  # Same cleanup path as RTSP - just needs .stop()

            relayed_video = relay.subscribe(video_track)

            processor_track = VideoProcessorTrack(
                relayed_video, session_vlm, text_callback=session_callback
            )

            pc.addTrack(processor_track)
            logger.info("Added video file processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create video file track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to open video file: {str(e)}"}),
            )
    else:
        # Webcam mode: wait for browser to send track
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")

            if track.kind == "video":
                # Create processor track with this session's VLM and session-scoped callback
                processor_track = VideoProcessorTrack(
                    relay.subscribe(track), session_vlm, text_callback=session_callback
                )

                # Add processed track back to connection
                pc.addTrack(processor_track)
                logger.info("Added processed video track back to peer connection")

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")

    # Handle offer
    await pc.setRemoteDescription(offer_sdp)

    # Create answer - this must happen after tracks are added
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"Created answer with {len(pc.getTransceivers())} transceivers")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def rtsp_start(request):
    """
    Start RTSP stream processing.

    Accepts RTSP URL and creates a video processing pipeline.

    POST /api/rtsp/start
    Body: {"rtsp_url": "rtsp://...", "session_id": "optional-id"}
    """
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url")
        session_id = data.get("session_id", "default")

        if not rtsp_url:
            logger.warning("RTSP start request missing rtsp_url")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Missing rtsp_url parameter"}),
            )

        # Check if session already exists
        if session_id in rtsp_tracks:
            logger.warning(f"RTSP session {session_id} already exists, stopping it first")
            await _stop_rtsp_session(session_id)

        logger.info(f"Starting RTSP stream for session {session_id}")

        # Create RTSP video track
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )

        # Create processor track with this session's VLM and session-scoped callback
        session = get_or_create_session(session_id)
        session_vlm = session["vlm_service"]
        session_callback = get_session_callback(session_id)
        processor_track = VideoProcessorTrack(
            rtsp_track, session_vlm, text_callback=session_callback
        )

        # Start background task to consume frames
        async def consume_frames():
            """Background task to continuously pull frames from processor track"""
            try:
                while not rtsp_track._stopped:
                    try:
                        _ = await processor_track.recv()
                        # Frame is processed, just discard it (VLM analysis happens in recv())
                    except StopAsyncIteration:
                        logger.info(f"RTSP stream {session_id} ended")
                        break
                    except Exception as e:
                        logger.error(f"Error consuming RTSP frame for {session_id}: {e}")
                        break
            finally:
                logger.info(f"Frame consumption stopped for {session_id}")

        frame_task = asyncio.create_task(consume_frames())

        # Store reference with frame task
        rtsp_tracks[session_id] = (rtsp_track, processor_track, frame_task)

        # Get stream stats
        stats = rtsp_track.get_stats()

        logger.info(
            f"RTSP stream started: {session_id} - {stats.get('codec')} "
            f"{stats.get('width')}x{stats.get('height')}"
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "started", "session_id": session_id, "stream_info": stats}),
        )

    except Exception as e:
        logger.error(f"Error starting RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_stop(request):
    """
    Stop RTSP stream processing.

    POST /api/rtsp/stop
    Body: {"session_id": "optional-id"}
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")

        await _stop_rtsp_session(session_id)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "stopped", "session_id": session_id}),
        )

    except Exception as e:
        logger.error(f"Error stopping RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_status(request):
    """
    Get status of all RTSP streams.

    GET /api/rtsp/status
    """
    try:
        status_list = []

        for session_id, (rtsp_track, processor_track, frame_task) in rtsp_tracks.items():
            stats = rtsp_track.get_stats()
            status_list.append(
                {
                    "session_id": session_id,
                    "connected": stats.get("connected"),
                    "frames_received": stats.get("frames_received"),
                    "stream_info": {
                        "codec": stats.get("codec"),
                        "width": stats.get("width"),
                        "height": stats.get("height"),
                        "fps": stats.get("fps"),
                    },
                }
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"active_streams": len(rtsp_tracks), "streams": status_list}),
        )

    except Exception as e:
        logger.error(f"Error getting RTSP status: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


# Directory where uploaded video files are stored - lives outside the git
# repo/package (in the user's cache dir) since uploaded content shouldn't be
# committed or bundled with the app
VIDEO_UPLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "live_vlm_webui", "uploaded_videos")

# Video files are typically much larger than the JSON/SDP bodies this server
# otherwise handles - cap uploads at 2GB to avoid accidentally filling disk
MAX_VIDEO_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


async def upload_video_file(request):
    """
    Accept a video file upload (multipart/form-data) and save it to disk so
    a VideoFileTrack can open it. Returns the saved path for use as
    video_file_path in a subsequent /offer request.

    POST /api/video/upload
    """
    try:
        os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

        reader = await request.multipart()
        field = await reader.next()

        if field is None or field.name != "file":
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Expected a multipart field named 'file'"}),
            )

        original_name = field.filename or "upload"
        ext = os.path.splitext(original_name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps(
                    {
                        "error": f"Unsupported file type '{ext}'. Allowed: "
                        f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
                    }
                ),
            )

        # Unique filename so concurrent/repeated uploads never collide
        saved_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(VIDEO_UPLOAD_DIR, saved_name)

        bytes_written = 0
        with open(saved_path, "wb") as f:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_VIDEO_UPLOAD_BYTES:
                    f.close()
                    os.remove(saved_path)
                    return web.Response(
                        status=413,
                        content_type="application/json",
                        text=json.dumps({"error": "File too large (max 2GB)"}),
                    )
                f.write(chunk)

        logger.info(f"Video file uploaded: {saved_path} ({bytes_written} bytes)")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {
                    "video_file_path": saved_path,
                    "original_filename": original_name,
                    "size_bytes": bytes_written,
                }
            ),
        )

    except Exception as e:
        logger.error(f"Error uploading video file: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def _stop_rtsp_session(session_id: str):
    """Helper function to stop an RTSP session"""
    if session_id in rtsp_tracks:
        rtsp_track, processor_track, frame_task = rtsp_tracks[session_id]

        # Cancel frame consumption task
        if frame_task and not frame_task.done():
            frame_task.cancel()
            try:
                await frame_task
            except asyncio.CancelledError:
                pass

        # Stop tracks
        try:
            processor_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping processor track: {e}")

        try:
            rtsp_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping RTSP track: {e}")

        # Remove from tracking
        del rtsp_tracks[session_id]
        logger.info(f"RTSP stream stopped: {session_id}")
    else:
        logger.warning(f"RTSP session {session_id} not found")


async def on_startup(app):
    """Initialize resources on server startup"""
    global gpu_monitor, gpu_monitor_task, yolo_detector, traffic_vlm_service

    # Initialize GPU monitor
    try:
        gpu_monitor = create_monitor()
        # Qualcomm boards: NPU busy % is our own NPU workloads' time (see
        # QualcommMonitor) - the YOLO detector's invoke() time plus the time a
        # VLM request is running on GenieX (Qwen3-VL, also on the NPU), capped
        # at 100%. Shown in /traffic's system stats.
        if hasattr(gpu_monitor, "npu_busy_source"):
            gpu_monitor.npu_busy_source = lambda: YoloDetector.npu_busy_seconds + VLMService.busy_seconds()
        logger.info("GPU monitor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitor: {e}")
        gpu_monitor = None

    # Start GPU monitoring background task
    if gpu_monitor:
        gpu_monitor_task = asyncio.create_task(gpu_monitor_loop())
        logger.info("GPU monitoring task started")

    # Load the /traffic cascade demo's detection model once at startup (HTP
    # delegate warmup is ~1-2s - see YoloDetector). A missing model/video
    # only disables /traffic; it must never take down the normal demo.
    if TRAFFIC_MODEL_PATH.exists() and TRAFFIC_LABELS_PATH.exists():
        try:
            yolo_detector = traffic_get_detector(TRAFFIC_MODEL_PATH.name)
            logger.info("Traffic detection model loaded - /traffic demo ready")
        except Exception as e:
            logger.error(f"Failed to load traffic detection model: {e}")
            yolo_detector = None
    else:
        logger.warning(
            f"Traffic detection model not found at {TRAFFIC_MODEL_PATH} - "
            "/traffic demo will be unavailable until it's exported (see demo_documentation.md)"
        )
        yolo_detector = None

    # Dedicated VLM client for /traffic click-to-caption - same GenieX
    # server as the main demo (default_vlm_config), but its own instance so
    # its cooldown/busy state never interferes with the main demo's.
    try:
        traffic_vlm_service = VLMService(
            model=default_vlm_config.get("model", "meta/llama-3.2-11b-vision-instruct"),
            api_base=default_vlm_config.get("api_base", "http://localhost:8000/v1"),
            api_key=default_vlm_config.get("api_key", "EMPTY"),
            prompt=TRAFFIC_CAPTION_DEFAULT_PROMPT,
            max_tokens=TRAFFIC_CAPTION_MAX_TOKENS,
        )
        logger.info("Traffic caption VLM service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize traffic caption VLM service: {e}")
        traffic_vlm_service = None


async def on_shutdown(app):
    """Cleanup on server shutdown"""
    global gpu_monitor, gpu_monitor_task

    logger.info("Shutting down server...")

    # Stop GPU monitoring task
    if gpu_monitor_task:
        gpu_monitor_task.cancel()
        try:
            await gpu_monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("GPU monitoring task stopped")

    # Cleanup GPU monitor
    if gpu_monitor:
        gpu_monitor.cleanup()
        logger.info("GPU monitor cleaned up")

    # Close all websockets and clear session state
    for ws in list(websockets):
        await ws.close()
    websockets.clear()
    session_websockets.clear()
    ws_to_session.clear()

    # Close /traffic demo websockets
    for ws in list(traffic_websockets):
        await ws.close()
    traffic_websockets.clear()

    # Close all RTSP streams
    for session_id in list(rtsp_tracks.keys()):
        await _stop_rtsp_session(session_id)
    logger.info("RTSP streams closed")

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    logger.info("Cleanup complete")


async def create_app(test_mode=False):
    """
    Create and configure the aiohttp web application.

    Args:
        test_mode: If True, skip GPU monitoring and use test configuration

    Returns:
        Configured web.Application instance
    """
    # Create web application
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/style.css", app_style)
    app.router.add_get("/app.js", app_script)
    app.router.add_get("/models", models)
    app.router.add_get("/detect-services", detect_services)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/offer", offer)

    # /traffic cascade demo - separate page/pipeline from the normal demo above
    app.router.add_get("/traffic", traffic_index)
    app.router.add_get("/traffic.css", traffic_style)
    app.router.add_get("/traffic.js", traffic_script)
    app.router.add_get("/api/traffic/ws", traffic_websocket_handler)
    app.router.add_post("/api/traffic/offer", traffic_offer)
    app.router.add_get("/api/traffic/settings", traffic_settings)
    app.router.add_post("/api/traffic/upload", traffic_upload)

    # RTSP endpoints
    app.router.add_post("/api/rtsp/start", rtsp_start)
    app.router.add_post("/api/rtsp/stop", rtsp_stop)
    app.router.add_get("/api/rtsp/status", rtsp_status)

    # Video file upload endpoint
    app.router.add_post("/api/video/upload", upload_video_file)

    # Serve static files (images, etc.)
    # Always serve from static/images within the package (works for both pip and dev installs)
    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images_dir = os.path.abspath(images_dir)

    if os.path.exists(images_dir):
        app.router.add_static("/images", images_dir, name="images")
        logger.info(f"Serving static files from: {images_dir}")
    else:
        logger.warning(f"⚠️  Static images directory not found: {images_dir}")

    # Serve self-hosted third-party JS libraries (lucide, marked, dompurify) -
    # previously loaded from unpkg.com/cdn.jsdelivr.net, which made the page
    # fail to fully initialize with no internet access (icons, markdown
    # rendering, and everything scripted after those tags in page order).
    vendor_dir = os.path.join(os.path.dirname(__file__), "static", "vendor")
    vendor_dir = os.path.abspath(vendor_dir)

    if os.path.exists(vendor_dir):
        app.router.add_static("/vendor", vendor_dir, name="vendor")
        logger.info(f"Serving vendored JS libraries from: {vendor_dir}")
    else:
        logger.warning(f"⚠️  Vendor JS directory not found: {vendor_dir}")

    # Serve favicon files
    favicon_dir = os.path.join(os.path.dirname(__file__), "static", "favicon")
    favicon_dir = os.path.abspath(favicon_dir)

    if os.path.exists(favicon_dir):
        app.router.add_static("/favicon", favicon_dir, name="favicon")
        logger.info(f"Serving favicon files from: {favicon_dir}")
    else:
        logger.warning(f"⚠️  Favicon directory not found: {favicon_dir}")

    if not test_mode:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)

    return app


def get_app_config_dir():
    """Get the application config directory following OS conventions"""
    import os
    from pathlib import Path

    # Follow XDG Base Directory spec on Linux, use OS-appropriate paths elsewhere
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "live-vlm-webui"
        else:
            # Linux/Unix (including Jetson)
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "live-vlm-webui"
            )
    else:
        # Windows
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "live-vlm-webui"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """Generate a self-signed SSL certificate if it doesn't exist"""
    import subprocess
    import os

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    import ssl
    from . import __version__

    parser = argparse.ArgumentParser(
        description="WebRTC Live VLM WebUI - Real-time vision model interaction",
        epilog="Examples:\n"
        "  vLLM:    python server.py --model llama-3.2-11b-vision-instruct --api-base http://localhost:8000/v1\n"
        "  SGLang:  python server.py --model llama-3.2-11b-vision-instruct --api-base http://localhost:30000/v1\n"
        "  Ollama:  python server.py --model llava:7b --api-base http://localhost:11434/v1\n"
        "  HTTPS:   python server.py --model llava:7b --api-base http://localhost:11434/v1 --ssl-cert cert.pem --ssl-key key.pem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to (default: 8090)")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find available port if default is taken",
    )
    parser.add_argument(
        "--model", help="VLM model name (optional, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--api-base", help="VLM API base URL (optional, will auto-detect or use NVIDIA NGC)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key - use 'EMPTY' for local servers, required for NVIDIA NGC/OpenAI (default: EMPTY)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe what you see in this image in one sentence.",
        help="Prompt to send to VLM (default: 'Describe what you see...')",
    )
    # Get default SSL cert paths (platform-specific)
    default_config_dir = get_app_config_dir()
    default_cert_path = str(default_config_dir / "cert.pem")
    default_key_path = str(default_config_dir / "key.pem")

    parser.add_argument("--process-every", type=int, default=30, help="Process every Nth frame")
    parser.add_argument(
        "--ssl-cert",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL certificate file (default: {default_cert_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL private key file (default: {default_key_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL (not recommended - webcam requires HTTPS)",
    )

    args = parser.parse_args()

    # Cloud deployment: env overrides for default API base, model, and frame interval
    if os.environ.get("LIVE_VLM_API_BASE"):
        if not args.api_base:
            args.api_base = os.environ.get("LIVE_VLM_API_BASE").strip()
            logger.info(f"Using API base from env: {args.api_base}")
    if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
        if not args.model:
            args.model = os.environ.get("LIVE_VLM_DEFAULT_MODEL").strip()
            logger.info(f"Using default model from env: {args.model}")
    if os.environ.get("LIVE_VLM_PROCESS_EVERY"):
        try:
            args.process_every = int(os.environ.get("LIVE_VLM_PROCESS_EVERY"))
            logger.info(f"Using process_every from env: {args.process_every}")
        except ValueError:
            pass

    # Set default SSL cert paths to config directory if not specified
    if args.ssl_cert is None:
        config_dir = get_app_config_dir()
        args.ssl_cert = str(config_dir / "cert.pem")
    if args.ssl_key is None:
        config_dir = get_app_config_dir()
        args.ssl_key = str(config_dir / "key.pem")

    # Auto-detect service and model if not specified
    api_base = args.api_base
    model = args.model
    api_key = args.api_key

    if not model or not api_base:
        logger.info("No model/API specified, auto-detecting local services...")
        detected_api_base, detected_model = asyncio.run(detect_local_service_and_model())

        if detected_api_base and detected_model:
            if not api_base:
                api_base = detected_api_base
            if not model:
                model = detected_model
        else:
            # Fall back to NVIDIA NGC
            logger.warning("⚠️  No local VLM service found (Ollama, vLLM, SGLang)")
            logger.info("📡 Falling back to NVIDIA API Catalog")
            logger.info("   You'll need an API key from: https://build.nvidia.com")
            if not api_base:
                api_base = "https://integrate.api.nvidia.com/v1"
            if not model:
                model = (
                    os.environ.get("LIVE_VLM_DEFAULT_MODEL") or "meta/llama-3.2-11b-vision-instruct"
                ).strip()
                if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
                    logger.info(f"Using default model from env: {model}")
            if api_key == "EMPTY":
                logger.warning("⚠️  API key required for NVIDIA API Catalog")
                logger.warning("   Set with: --api-key YOUR_API_KEY")
                logger.warning("   Or use WebUI to configure API settings after starting")

    # Initialize VLM service and default session for multi-session support
    global vlm_service, default_vlm_config
    vlm_service = VLMService(model=model, api_base=api_base, api_key=api_key, prompt=args.prompt)
    default_vlm_config = {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "prompt": args.prompt,
    }
    sessions["default"] = {
        "vlm_service": vlm_service,
        "show_request_payload": False,
        "show_response_payload": False,
    }

    # Log initialization with better formatting
    service_name = "Local" if "localhost" in api_base or "127.0.0.1" in api_base else "Cloud"
    logger.info("Initialized VLM service:")
    logger.info(f"  Model: {model}")
    logger.info(f"  API: {api_base} ({service_name})")
    logger.info(f"  Prompt: {args.prompt}")

    # Update frame processing rate in VideoProcessorTrack if needed
    # (This is a bit hacky but works for this demo)
    VideoProcessorTrack.process_every_n_frames = args.process_every

    # Create web application using create_app
    app = asyncio.run(create_app(test_mode=False))

    # Setup SSL (auto-generate certificates if needed)
    ssl_context = None
    protocol = "http"
    if not args.no_ssl:
        # Try to auto-generate if certificates don't exist
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            success = generate_self_signed_cert(args.ssl_cert, args.ssl_key)
            if not success:
                # FAIL FAST - SSL is required for webcam access
                logger.error("")
                logger.error("❌ Cannot start server without SSL certificates")
                logger.error("❌ Webcam access requires HTTPS!")
                logger.error("")
                logger.error("🔧 To fix, install openssl:")
                logger.error("   Linux/Jetson: sudo apt install openssl")
                logger.error("   macOS: brew install openssl")
                logger.error("")
                logger.error("   Then restart the server")
                logger.error("")
                logger.error(
                    "⚠️  Or run with --no-ssl if you don't need camera access (not recommended)"
                )
                logger.error("")
                sys.exit(1)

        # Load certificates (they must exist at this point)
        if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
            protocol = "https"
            logger.info("SSL enabled - using HTTPS")
        else:
            # This should never happen, but just in case
            logger.error("❌ SSL certificates missing after generation - unexpected error")
            sys.exit(1)
    else:
        logger.warning("⚠️  SSL disabled with --no-ssl flag")
        logger.warning("⚠️  Webcam access will NOT work without HTTPS!")

    # Get network addresses
    import socket
    import subprocess

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Access the server at:")
    logger.info(f"  Local:   {protocol}://localhost:{args.port}")

    # Get network interfaces - try multiple methods for cross-platform support
    network_ips = []

    # Method 1: hostname -I (Linux)
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                # Filter out loopback and docker bridges (172.17.x.x)
                if not ip.startswith("127.") and not ip.startswith("172.17."):
                    network_ips.append(ip)
    except Exception:
        pass

    # Method 2: Socket method (cross-platform fallback)
    if not network_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1":
                network_ips.append(ip)
        except Exception:
            pass

    # Display all found network IPs
    for ip in network_ips:
        logger.info(f"  Network: {protocol}://{ip}:{args.port}")

    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


def stop():
    """Stop the running live-vlm-webui server"""
    import sys
    import time

    try:
        import psutil
    except ImportError:
        logger.error("psutil is required for the stop command")
        logger.error("Install it with: pip install live-vlm-webui[dev]")
        sys.exit(1)

    print("Stopping Live VLM WebUI server...")

    # Find and kill processes running live_vlm_webui.server
    found = False
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    # Don't kill the stop command itself
                    if "stop" not in cmdline_str:
                        found = True
                        print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                        proc.terminate()
                        killed.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("✓ No running server found")
        return

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for proc in killed:
        try:
            if proc.is_running():
                print(f"  Force killing process {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Final verification
    time.sleep(1)
    still_running = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    if "stop" not in cmdline_str:
                        still_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if still_running:
        print("❌ Failed to stop server")
        sys.exit(1)
    else:
        print("✓ Server stopped successfully")


if __name__ == "__main__":
    main()
