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
Vehicle Detection Module

Wraps a YOLO26n TFLite model - exported via Qualcomm AI Hub for the
Dragonwing IQ-9075 EVK (`qai-hub-models export yolo26_det --device
"Dragonwing IQ-9075 EVK" --runtime tflite --precision float`) - running on
the Hexagon NPU through the QNN HTP delegate. Detection is cheap enough
(single-digit ms/frame on the NPU) to run continuously; the VLM is the
actual bottleneck (~2.4s/call), so this module only feeds it cropped
regions on demand rather than every frame.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

logger = logging.getLogger(__name__)

# Default class filter for the traffic-monitoring demo: only report COCO
# classes relevant to a road/intersection scene, so the overlay/caption
# targets stay meaningful instead of flagging every pedestrian, bench, etc.
DEFAULT_TARGET_CLASSES = {"car", "motorcycle", "bus", "truck"}

QNN_HTP_DELEGATE = "libQnnTFLiteDelegate.so"


@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame pixel coords


class YoloDetector:
    """YOLO26 object detector for the QCS9075/IQ-9075 Hexagon NPU.

    The exported graph already decodes the raw anchor grid into per-box
    (xyxy coordinates in 640x640 model-input space, best-class score, class
    index) - see qai_hub_models' `yolo_detect_postprocess` - but does NOT
    run NMS, so the raw model output contains many overlapping/duplicate
    boxes per real object. Confidence thresholding and per-class NMS are
    done here via cv2.dnn.NMSBoxes.
    """

    INPUT_SIZE = 640  # fixed at export time; this model was compiled for 640x640 only

    # Cumulative invoke() time on the Hexagon NPU, process-wide. The system
    # monitor turns it into an NPU busy % for /traffic's telemetry - the NPU has
    # no readable utilization counter. Only touched from the detection thread.
    npu_busy_seconds: float = 0.0

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        backend: str = "htp",
        conf_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        target_classes: Optional[Set[str]] = None,
    ):
        """
        Args:
            model_path: Path to the exported yolo26n .tflite file.
            labels_path: Path to a newline-separated COCO class labels file
                (index order must match the model's class_idx output).
            backend: "htp" to run on the Hexagon NPU via QNN delegate
                (falls back to CPU with a warning if the delegate fails to
                load), or "cpu" to force CPU/XNNPACK.
            conf_threshold: Minimum class score to keep a candidate box.
            nms_iou_threshold: IoU threshold for suppressing overlapping
                boxes of the same class.
            target_classes: Set of label strings to report; other classes
                are detected but filtered out. Defaults to
                DEFAULT_TARGET_CLASSES (vehicle classes only).
        """
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.target_classes = (
            target_classes if target_classes is not None else DEFAULT_TARGET_CLASSES
        )

        with open(labels_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]

        self.backend = backend
        delegates = []
        if backend == "htp":
            try:
                start = time.perf_counter()
                delegates = [load_delegate(QNN_HTP_DELEGATE, options={"backend_type": "htp"})]
                logger.info(
                    f"QNN HTP delegate loaded in {time.perf_counter() - start:.2f}s "
                    "- detection will run on the Hexagon NPU"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load QNN HTP delegate ({e}) - falling back to CPU. "
                    "Detection will be far slower than on the NPU; confirm "
                    "libQnnTFLiteDelegate.so is on the library path and the "
                    "board's QNN SDK is installed."
                )
                self.backend = "cpu"
                delegates = []

        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self._input_index = input_details[0]["index"]
        self._output_indices = {d["name"]: d["index"] for d in output_details}

        input_shape = tuple(input_details[0]["shape"])
        expected_shape = (1, self.INPUT_SIZE, self.INPUT_SIZE, 3)
        if input_shape != expected_shape:
            raise ValueError(
                f"Unexpected model input shape {input_shape}, expected "
                f"{expected_shape} (NHWC float32, matching the float TFLite "
                "export). A model exported with different precision (e.g. "
                "w8a16) may use a different layout - check "
                "interpreter.get_input_details() and update preprocessing "
                "rather than assuming this shape holds."
            )

        logger.info(
            f"YoloDetector ready: {Path(model_path).name} on {self.backend}, "
            f"{len(self.labels)} classes, filtering to {sorted(self.target_classes)}"
        )

    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Resize frame to fit within INPUT_SIZE x INPUT_SIZE preserving aspect
        ratio, centered on a black canvas - matches the resize_pad preprocessing
        qai_hub_models uses when validating this model, so decoded box
        coordinates map back to the original frame correctly."""
        h, w = frame.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.INPUT_SIZE - new_w) // 2
        pad_y = (self.INPUT_SIZE - new_h) // 2
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        """Run detection on a single RGB frame (H, W, 3), uint8.

        Returns detections filtered to self.target_classes, in original
        frame pixel coordinates, after confidence thresholding and NMS.
        """
        orig_h, orig_w = rgb_frame.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(rgb_frame)

        input_tensor = (canvas.astype(np.float32) / 255.0)[np.newaxis, ...]
        self.interpreter.set_tensor(self._input_index, input_tensor)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        if self.backend == "htp":
            YoloDetector.npu_busy_seconds += time.perf_counter() - t0

        # boxes: [8400, 4] xyxy in 640x640 model-input space; scores/class_idx: [8400]
        boxes = self.interpreter.get_tensor(self._output_indices["boxes"])[0]
        scores = self.interpreter.get_tensor(self._output_indices["scores"])[0]
        class_idx = self.interpreter.get_tensor(self._output_indices["class_idx"])[0]

        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []
        boxes, scores, class_idx = boxes[keep], scores[keep], class_idx[keep]

        # cv2.dnn.NMSBoxes wants [x, y, w, h] rects, not xyxy
        nms_boxes = np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
            axis=1,
        )

        detections: List[Detection] = []
        for cls in np.unique(class_idx):
            label = self.labels[int(cls)] if int(cls) < len(self.labels) else str(int(cls))
            if label not in self.target_classes:
                continue

            cls_mask = class_idx == cls
            idxs = cv2.dnn.NMSBoxes(
                nms_boxes[cls_mask].tolist(),
                scores[cls_mask].tolist(),
                self.conf_threshold,
                self.nms_iou_threshold,
            )
            if len(idxs) == 0:
                continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            for i in np.array(idxs).flatten():
                x1, y1, x2, y2 = cls_boxes[i]
                # Undo letterbox: remove the pad offset, then rescale to the original frame
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                detections.append(
                    Detection(
                        label=label,
                        score=float(cls_scores[i]),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        return self._dedupe_cross_class(detections)

    def _dedupe_cross_class(
        self, detections: List[Detection], iou_threshold: float = 0.5
    ) -> List[Detection]:
        """cv2.dnn.NMSBoxes above only suppresses overlapping boxes within
        the SAME class, so a vehicle the model is genuinely torn on (e.g.
        borderline car/truck) can survive NMS as two overlapping boxes with
        different labels on the same physical object - confirmed via direct
        diagnostic (frame with `car score=0.47` and `truck score=0.53` at
        near-identical coordinates). This final class-agnostic pass keeps
        only the single highest-confidence box per overlapping region,
        regardless of label, which is what a downstream tracker needs to
        treat one physical object as one track."""
        ordered = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: List[Detection] = []
        for d in ordered:
            if all(self._iou(d.bbox, k.bbox) < iou_threshold for k in kept):
                kept.append(d)
        return kept

    @staticmethod
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


if __name__ == "__main__":
    # Standalone Phase 1 smoke test: run against tests/fixtures/bus.jpg,
    # confirm the NPU delegate loads, and print detections + latency.
    #   .venv/bin/python -m live_vlm_webui.detector [image_path] [--all-classes]
    import argparse

    parser = argparse.ArgumentParser(description="Standalone YoloDetector smoke test")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bus.jpg"),
        help="Path to a test image (default: tests/fixtures/bus.jpg)",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[2] / "models" / "yolo26n_det_qcs9075.tflite"),
    )
    parser.add_argument(
        "--labels",
        default=str(Path(__file__).resolve().parents[2] / "models" / "coco_labels.txt"),
    )
    parser.add_argument("--backend", default="htp", choices=["htp", "cpu"])
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Report all COCO classes instead of just the vehicle-class filter",
    )
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    target_classes = None if args.all_classes else DEFAULT_TARGET_CLASSES
    detector = YoloDetector(
        model_path=args.model,
        labels_path=args.labels,
        backend=args.backend,
        conf_threshold=args.conf,
        target_classes=target_classes,
    )

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # First call includes any one-time delegate warmup; run a couple more to
    # get a representative steady-state latency.
    for i in range(3):
        start = time.perf_counter()
        dets = detector.detect(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"Run {i + 1}: {elapsed_ms:.1f} ms, {len(dets)} detections")

    print(f"\n{args.image} -> {len(dets)} detections (backend={detector.backend}):")
    for d in dets:
        print(f"  {d.label:12s} score={d.score:.2f} bbox={d.bbox}")

    out_path = str(Path(args.image).with_suffix("")) + "_annotated.jpg"
    annotated = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{d.label} {d.score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(out_path, annotated)
    print(f"Annotated image saved to {out_path}")
