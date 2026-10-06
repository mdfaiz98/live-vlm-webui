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
Lightweight IoU-based vehicle tracker for the /traffic cascade demo.

YOLO26n has no notion of object identity across frames - every frame's
detections are independent. Without tracking, the same physical vehicle
gets reported as a "new" detection on every frame it's visible, which is
wrong for a history panel (the same car would appear dozens of times).

This is deliberately NOT a general-purpose multi-object tracker (no Kalman
filter, no re-identification after occlusion) - just enough greedy
IoU-matching continuity to dedupe a fixed-camera highway scene with a
handful of vehicles in frame at once, so each vehicle is logged to the
history panel once.

It also tracks each vehicle's largest-observed bounding box (`best_bbox`)
and delays reporting until that box has passed its peak size - a vehicle
entering frame is typically small/partial (sometimes just a plate-height
sliver at the frame edge) for its first few confirmed frames, so cropping
at first-confirmation produces a poor thumbnail. Waiting for the peak (or
for the vehicle to leave frame, as a fallback) gives the fullest view.
"""

import uuid
from typing import List, Optional

import numpy as np

from .detector import Detection


def _iou(box_a, box_b) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


def _area(bbox) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


class Track:
    def __init__(self, label: str, bbox: tuple, score: float):
        self.id = uuid.uuid4().hex[:8]
        self.label = label
        self.bbox = bbox
        self.score = score
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.emitted = False
        # Best-observed-so-far, used for the reported crop/label/score -
        # deliberately separate from the "current" fields above, which track
        # whatever the model reports this frame (used for IoU matching).
        # best_crop is captured as actual pixels AT THE MOMENT the peak is
        # recorded (not re-cropped later from a since-moved-on frame).
        self.best_bbox = bbox
        self.best_score = score
        self.best_area = _area(bbox)
        self.best_crop: Optional[np.ndarray] = None
        self.shrink_count = 0


class SimpleVehicleTracker:
    """Feed per-frame detections in via update(); get back the list of
    tracks ready to report to the history panel this call - each track is
    returned at most once, using its best (largest) observed bbox/score."""

    IOU_MATCH_THRESHOLD = 0.3
    CONFIRM_HITS = 3  # consecutive matched frames before a track counts as real (filters noise)
    SHRINK_CONFIRM_FRAMES = 5  # consecutive frames past peak size before reporting
    MAX_MISSES = 15  # frames a track can go unmatched before being dropped

    def __init__(self):
        self.tracks: List[Track] = []
        # None: a frame "shrinks" when the box is at all smaller than its peak.
        # A ratio (e.g. 0.8): only when it's below ratio * peak - so a car that
        # stops at a parking barrier (box jitters around one size) isn't
        # reported until it actually drives on, and its reported crop is its
        # closest view, not the one with the barrier arm across the plate.
        self.shrink_ratio: Optional[float] = None
        # Entry-zone mode (parking entrance), on top of shrink_ratio:
        # - a box cut off by the frame edge never becomes the best crop (a car
        #   half out of the frame gave a thin useless strip),
        # - the first sighting can be the best crop (a car leaving is biggest
        #   when it first appears, so "a later, bigger box" never happened and
        #   it was reported with no crop at all),
        # - a vehicle with no usable crop is not reported.
        # Off by default, so other videos keep the original behaviour.
        self.zone_mode = False
        self.EDGE_MARGIN = 0.01  # fraction of the frame: a box this close to an edge counts as cut off

    def update(self, detections: List[Detection], img: np.ndarray) -> List[Track]:
        """Feed one frame's detections in, along with that same frame's
        image (needed to capture pixel crops at the exact moment a track
        hits a new peak size - capturing bbox coordinates alone and
        re-cropping later would crop the wrong, since-moved-on frame)."""
        frame_h, frame_w = img.shape[:2]
        unmatched = list(detections)
        ready: List[Track] = []

        for track in self.tracks:
            # Match by spatial overlap only, not label - the classifier's
            # label for the same physical vehicle can flicker frame to frame
            # (e.g. car/truck) at these confidence levels.
            best_iou, best_det = 0.0, None
            for det in unmatched:
                score = _iou(det.bbox, track.bbox)
                if score > best_iou:
                    best_iou, best_det = score, det

            if best_det is None or best_iou < self.IOU_MATCH_THRESHOLD:
                track.misses += 1
                continue

            track.bbox = best_det.bbox
            track.label = best_det.label  # use the most recent classification
            track.score = best_det.score
            track.hits += 1
            track.misses = 0
            unmatched.remove(best_det)

            if not track.confirmed and track.hits >= self.CONFIRM_HITS:
                track.confirmed = True

            area = _area(best_det.bbox)
            if self.zone_mode:
                usable = not self._cut_off(best_det.bbox, frame_w, frame_h)
                better = usable and (area >= track.best_area or track.best_crop is None)
            else:
                better = area >= track.best_area
            if better:
                track.best_bbox = best_det.bbox
                track.best_score = best_det.score
                track.best_area = area
                track.shrink_count = 0

                x1, y1, x2, y2 = best_det.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                if x2 > x1 and y2 > y1:
                    track.best_crop = img[y1:y2, x1:x2].copy()
            elif self.shrink_ratio is None or area < track.best_area * self.shrink_ratio:
                track.shrink_count += 1

            if (
                track.confirmed
                and not track.emitted
                and track.shrink_count >= self.SHRINK_CONFIRM_FRAMES
                and not (self.zone_mode and track.best_crop is None)
            ):
                track.emitted = True
                ready.append(track)

        still_alive: List[Track] = []
        for track in self.tracks:
            if track.misses > self.MAX_MISSES:
                # About to be dropped (e.g. exited frame) without ever
                # completing a clean shrink-after-peak - report it anyway
                # using whatever the best crop we did see was, rather than
                # losing the sighting entirely.
                if track.confirmed and not track.emitted and not (self.zone_mode and track.best_crop is None):
                    track.emitted = True
                    ready.append(track)
            else:
                still_alive.append(track)
        self.tracks = still_alive

        for det in unmatched:
            track = Track(label=det.label, bbox=det.bbox, score=det.score)
            if self.zone_mode:
                if self._cut_off(det.bbox, frame_w, frame_h):
                    track.best_area = 0  # let the first whole view become the best crop
                else:
                    x1, y1, x2, y2 = det.bbox
                    crop = img[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                    if crop.size:
                        track.best_crop = crop.copy()
            self.tracks.append(track)

        return ready

    def _cut_off(self, bbox, frame_w: int, frame_h: int) -> bool:
        x1, y1, x2, y2 = bbox
        mx, my = self.EDGE_MARGIN * frame_w, self.EDGE_MARGIN * frame_h
        return x1 <= mx or y1 <= my or x2 >= frame_w - mx or y2 >= frame_h - my
