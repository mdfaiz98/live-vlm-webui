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
Detection Video Track
Wraps an incoming video track and, once enabled (see detection_enabled -
starts off, the demo plays plain video until a UI button turns it on),
runs YoloDetector on every frame and draws bounding boxes directly onto
the frame before it's sent over WebRTC - for the /traffic cascade demo.
Boxes are burned into the video itself (not a separate client-side canvas
overlay), matching the annotated-video style already validated against
this same clip (see annotate_video.py smoke test).

Separately, a lightweight IoU tracker (see tracker.py) dedupes per-frame
detections so each physical vehicle is reported via callback exactly once
- when first confirmed - rather than once per frame it's visible, for the
"Detected Vehicles" history panel. This never calls the VLM itself;
per-object captioning is triggered separately (on-demand, Phase 3) against
these same crops.
"""

import asyncio
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import av
import cv2
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

from .detector import Detection, YoloDetector
from .tracker import SimpleVehicleTracker, Track

logger = logging.getLogger(__name__)

BOX_COLOR = (35, 226, 138)  # BGR - matches the UI's --ring-cpu-color green
LABEL_TEXT_COLOR = (10, 12, 10)
ZONE_COLOR = (0, 200, 255)  # BGR amber - distinct from the green vehicle boxes
CROP_THUMBNAIL_MAX_DIM = 160  # small panel-list thumbnail
CROP_MODAL_MAX_DIM = 480  # larger crop for the click-to-open detail modal
CROP_JPEG_QUALITY = 80

# The TFLite/QNN HTP interpreter backing YoloDetector is a single shared
# resource (one NPU, one Interpreter instance reused across every /traffic
# connection) and is not safe for concurrent invoke() calls from multiple
# threads. A dedicated single-worker executor serializes all detection calls
# process-wide, regardless of how many DetectionVideoTrack instances exist
# (e.g. from rapid page reloads overlapping during reconnect) - the default
# asyncio executor has multiple worker threads and would not guarantee this.
_detection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-detect")


class DetectionVideoTrack(VideoStreamTrack):
    """Video track that draws detection boxes onto every frame and, on a
    throttled interval, reports cropped thumbnails via a callback.

    Detection starts OFF (see detection_enabled) - the demo plays the plain
    video immediately on connect, and detection only turns on once
    enable_detection() is called (from a UI button, via
    traffic_websocket_handler's "toggle_detection" message). While off, no
    NPU inference happens at all - not just "boxes hidden" - frames pass
    through untouched, same as before detection existed on this page."""

    def __init__(
        self,
        track: VideoStreamTrack,
        detector: YoloDetector,
        detection_callback: Optional[
            Callable[[List[Track], List[Optional[str]], List[Optional[str]], int, int], None]
        ] = None,
    ):
        super().__init__()
        self.track = track
        self.detector = detector
        self.detection_callback = detection_callback
        self.tracker = SimpleVehicleTracker()
        self.frame_count = 0
        self.detection_enabled = False
        # Optional entry zone (x1, y1, x2, y2 as fractions of the frame): only
        # vehicles whose box bottom-centre - where the wheels touch the ground -
        # is inside it are drawn and logged - e.g. the
        # barrier lanes of a parking entrance, not cars passing on the street
        # behind it. None = the whole frame.
        self.entry_zone: Optional[Tuple[float, float, float, float]] = None

    def set_entry_zone(self, zone: Optional[Tuple[float, float, float, float]]):
        """Entry zone + its reporting rules: with a zone (a parking entrance),
        a vehicle is logged once it has clearly moved on from its closest point
        (box < 80% of its peak), not while it waits at the barrier, and its
        crop is its biggest whole (not cut-off) view - see SimpleVehicleTracker."""
        self.entry_zone = zone
        self.tracker.shrink_ratio = 0.8 if zone is not None else None
        self.tracker.zone_mode = zone is not None

    def enable_detection(self):
        if not self.detection_enabled:
            self.detection_enabled = True
            logger.info("[traffic] Detection enabled")

    def disable_detection(self):
        if self.detection_enabled:
            self.detection_enabled = False
            logger.info("[traffic] Detection disabled")

    async def recv(self):
        try:
            frame = await self.track.recv()
            self.frame_count += 1

            if not self.detection_enabled:
                return frame

            # BGR matches both cv2's drawing conventions and av.VideoFrame's
            # "bgr24" format, so no extra channel-order conversion is needed
            # on the way in or back out.
            img = frame.to_ndarray(format="bgr24")

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # detector expects RGB
            start = time.perf_counter()
            loop = asyncio.get_event_loop()
            # Blocking (~15ms) TFLite/HTP inference - run off the event loop
            # so it doesn't stall WebRTC packetization or other sessions, and
            # on the dedicated single-worker executor so concurrent
            # connections can't invoke() the shared interpreter at once.
            detections = await loop.run_in_executor(_detection_executor, self.detector.detect, rgb)
            latency_ms = (time.perf_counter() - start) * 1000
            if self.entry_zone is not None:
                detections = self._in_zone(detections, img.shape[1], img.shape[0])

            if self.frame_count % 90 == 0:
                logger.debug(
                    f"Detection frame {self.frame_count}: {len(detections)} objects, "
                    f"{latency_ms:.1f}ms"
                )

            annotated = self._draw_boxes(img, detections)

            ready_tracks = self.tracker.update(detections, img)
            if self.detection_callback and ready_tracks:
                crops, crops_large = self._encode_crops(ready_tracks)
                h, w = img.shape[:2]
                self.detection_callback(ready_tracks, crops, crops_large, w, h)

            new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except MediaStreamError:
            logger.debug("Detection video track ended")
            raise
        except Exception as e:
            logger.error(f"Error processing detection frame: {e}", exc_info=True)
            raise

    def reset_tracker(self):
        """Forget all tracked vehicles (keeps the tracker's settings) - called
        when an entry-zone video loops, see server.traffic_offer."""
        self.tracker.tracks = []

    def _zone_px(self, w: int, h: int) -> Tuple[int, int, int, int]:
        zx1, zy1, zx2, zy2 = self.entry_zone
        return int(zx1 * w), int(zy1 * h), int(zx2 * w), int(zy2 * h)

    def _in_zone(self, detections: List[Detection], w: int, h: int) -> List[Detection]:
        x1, y1, x2, y2 = self._zone_px(w, h)
        return [
            d for d in detections
            if x1 <= (d.bbox[0] + d.bbox[2]) / 2 <= x2 and y1 <= d.bbox[3] <= y2
        ]

    def _draw_zone(self, img: np.ndarray) -> None:
        """Entry zone: light translucent fill, outline and an "ENTRY ZONE" tag."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self._zone_px(w, h)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), ZONE_COLOR, -1)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, dst=img)
        cv2.rectangle(img, (x1, y1), (x2, y2), ZONE_COLOR, 3)
        (tw, th), base = cv2.getTextSize("ENTRY ZONE", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 16, y1 + th + base + 12), ZONE_COLOR, -1)
        cv2.putText(img, "ENTRY ZONE", (x1 + 8, y1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    def _draw_boxes(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated = img.copy()
        if self.entry_zone is not None:
            self._draw_zone(annotated)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 4)

            label_text = f"{d.label} {d.score * 100:.0f}%"
            font_scale, thickness = 1.3, 3
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_y = max(text_h + baseline + 6, y1)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - baseline - 6),
                (x1 + text_w + 10, label_y),
                BOX_COLOR,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                LABEL_TEXT_COLOR,
                thickness,
            )
        return annotated

    def _encode_crops(
        self, tracks: List[Track]
    ) -> tuple[List[Optional[str]], List[Optional[str]]]:
        """Resize + JPEG/base64 encode each track's already-captured best_crop
        (pixels grabbed at the moment tracker.update() saw that track's peak
        size, on whichever frame that was - not re-cropped here) at two
        sizes: a small thumbnail for the panel list, and a larger version
        for the click-to-open detail modal. Both are still well below the
        full-res crop the VLM itself sees (see server.py's
        traffic_track_store) - the modal is for a human to look at, not to
        match the VLM's input exactly."""
        thumbs: List[Optional[str]] = []
        larges: List[Optional[str]] = []
        for t in tracks:
            if t.best_crop is None or t.best_crop.size == 0:
                thumbs.append(None)
                larges.append(None)
                continue
            thumbs.append(self._resize_and_encode(t.best_crop, CROP_THUMBNAIL_MAX_DIM))
            larges.append(self._resize_and_encode(t.best_crop, CROP_MODAL_MAX_DIM))

        return thumbs, larges

    @staticmethod
    def _resize_and_encode(crop: np.ndarray, max_dim: int) -> Optional[str]:
        ch, cw = crop.shape[:2]
        scale = max_dim / max(ch, cw)
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
        return base64.b64encode(buf).decode("ascii") if ok else None
