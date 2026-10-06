"""SimpleVehicleTracker: default behaviour vs entry-zone (parking) mode."""

import numpy as np

from live_vlm_webui.detector import Detection
from live_vlm_webui.tracker import SimpleVehicleTracker

W, H = 1920, 1080


def det(x1, y1, x2, y2):
    return Detection(label="car", score=0.9, bbox=(x1, y1, x2, y2))


def frame(value=0):
    return np.full((H, W, 3), value, np.uint8)


def zone_tracker():
    t = SimpleVehicleTracker()
    t.shrink_ratio, t.zone_mode = 0.8, True
    return t


def feed(tracker, boxes, start_value=0):
    """Feed one box per frame (frame pixel value = frame index, to identify crops)."""
    out = []
    for i, b in enumerate(boxes):
        out += [(i, r) for r in tracker.update([det(*b)] if b else [], frame(start_value + i))]
    return out


def test_waiting_at_barrier_not_logged_until_car_drives_on():
    # approach, then wait ~4 s with the box jittering a few % below its peak,
    # then drive on toward the camera (grows smoothly), then leave (shrinks)
    approach = [(800, 300, 1000 + 10 * i, 500 + 10 * i) for i in range(10)]
    wait = [(800, 300, 1080 + (i % 3) * 4, 580 + (i % 3) * 4) for i in range(60)]
    drive_on = [(800 - 10 * i, 300 + 5 * i, 1084 + 12 * i, 584 + 12 * i) for i in range(20)]
    last = drive_on[-1]
    leave = [(last[0] + 8 * i, last[1] + 4 * i, last[2] - 16 * i, last[3] - 12 * i) for i in range(15)]
    seq = approach + wait + drive_on + leave + [None] * 20
    default = feed(SimpleVehicleTracker(), seq)
    zoned = feed(zone_tracker(), seq)
    # default mode reports the car while it is still waiting (crop with the barrier arm in front);
    # zone mode reports it once, only after it has driven on, with its closest view
    assert default and default[0][0] < len(approach) + len(wait)
    assert len(zoned) == 1 and zoned[0][0] >= len(approach) + len(wait) + len(drive_on)
    closest = len(approach) + len(wait) + len(drive_on)  # leave[0] repeats the last drive_on box; ties keep the newer frame
    assert int(zoned[0][1].best_crop[0, 0, 0]) in (closest - 1, closest)


def test_first_sighting_is_the_crop_for_a_car_leaving():
    # a car that is biggest when first seen (driving away from the camera)
    leaving = [(400, 300, 1400 - 40 * i, 1000 - 30 * i) for i in range(20)]
    default = feed(SimpleVehicleTracker(), leaving + [None] * 20)
    zoned = feed(zone_tracker(), leaving + [None] * 20)
    assert default and default[0][1].best_crop is None  # the original gap
    assert zoned and zoned[0][1].best_crop is not None
    assert int(zoned[0][1].best_crop[0, 0, 0]) == 0  # crop from the very first frame


def test_cut_off_boxes_never_become_the_crop():
    # a car half out of the bottom of the frame, then fully visible but smaller
    cut = [(200, 900, 1200, H - 2) for _ in range(5)]
    whole = [(300, 500, 1000, 900 - 10 * i) for i in range(20)]
    zoned = feed(zone_tracker(), cut + whole + [None] * 20)
    assert zoned
    crop = zoned[0][1].best_crop
    assert crop is not None and crop.shape[0] < 900 - 500 + 1  # from a whole view, not the cut-off strip
    assert int(crop[0, 0, 0]) >= len(cut)


def test_no_card_without_a_crop_in_zone_mode():
    # only ever seen cut off by the frame edge -> nothing worth showing
    cut_only = [(0, 600, 900, 1000) for _ in range(10)]
    assert feed(zone_tracker(), cut_only + [None] * 20) == []
