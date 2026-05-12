"""
main.py
=======
Live stereo viewer.

Opens both CSI cameras, loads data/stereo_calib.npz, and shows the rectified
side-by-side feed with horizontal epipolar lines — same display style as
`calibration/stereo_calibrate.py --mode verify`.

Press Q to quit, R to toggle raw/rectified view, L to toggle epipolar lines.
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils import CameraConfig, DualCameraConfig, DualCSICamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("main")

# Must match the rotations used during stereo calibration.
_CAM0_ROTATION = 270
_CAM1_ROTATION = 90

WIDTH         = 1640
HEIGHT        = 1232
FPS           = 30
SENSOR_MODE   = 3
FLIP_METHOD   = 0
DISPLAY_SCALE = 0.4
CALIB_PATH    = Path("data/stereo_calib.npz")

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 220, 100)
_AMBER = (0, 180, 240)
_WHITE = (230, 230, 230)


def _text(img, txt, xy, scale=0.55, color=_WHITE, thickness=1):
    cv2.putText(img, txt, xy, _FONT, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, _FONT, scale, color,       thickness,     cv2.LINE_AA)


def _load_calib(path: Path):
    if not path.exists():
        log.error("Stereo calibration not found: %s — run calibration/stereo_calibrate.py first",
                  path)
        sys.exit(1)
    d = np.load(str(path))
    return {
        "map0x":    d["map0x"],
        "map0y":    d["map0y"],
        "map1x":    d["map1x"],
        "map1y":    d["map1y"],
        "rms":      float(d["rms"][0]),
        "baseline": float(np.linalg.norm(d["T"])) * 1000.0,  # mm
        "img_size": tuple(d["img_size"].tolist()),
    }


def _build_dual_cfg() -> DualCameraConfig:
    cam_kwargs = dict(width=WIDTH, height=HEIGHT, framerate=FPS,
                      flip_method=FLIP_METHOD, sensor_mode=SENSOR_MODE)
    cfg      = DualCameraConfig()
    cfg.cam0 = CameraConfig(sensor_id=0, rotation=_CAM0_ROTATION, **cam_kwargs)
    cfg.cam1 = CameraConfig(sensor_id=1, rotation=_CAM1_ROTATION, **cam_kwargs)
    return cfg


def main() -> None:
    calib = _load_calib(CALIB_PATH)
    log.info("Loaded %s  RMS=%.4f px  baseline=%.1f mm  img_size=%s",
             CALIB_PATH.name, calib["rms"], calib["baseline"], calib["img_size"])

    log.info("Opening both CSI cameras …")
    with DualCSICamera(_build_dual_cfg()) as cams:
        # Warm-up
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            f0, f1 = cams.read()
            if f0 is not None and f1 is not None:
                break
            time.sleep(0.05)
        else:
            log.error("Cameras did not produce frames within 8 s")
            sys.exit(1)

        log.info("Live view — Q=quit  R=toggle raw/rectified  L=toggle epipolar lines")

        show_rectified = True
        show_lines     = True

        while True:
            try:
                f0, f1 = cams.read_sync(timeout=1.0, max_dt=0.020)
            except TimeoutError:
                continue

            if show_rectified:
                view0 = cv2.remap(f0, calib["map0x"], calib["map0y"], cv2.INTER_LINEAR)
                view1 = cv2.remap(f1, calib["map1x"], calib["map1y"], cv2.INTER_LINEAR)
                mode_label = "rectified"
            else:
                view0 = f0
                view1 = f1
                mode_label = "raw"

            h, w = view0.shape[:2]
            combined = np.hstack([view0, view1])

            if show_rectified and show_lines:
                for y in range(0, h, 60):
                    cv2.line(combined, (0, y), (w * 2, y), (0, 200, 200), 1)

            _text(combined, f"cam0 {mode_label}", (10, 26), color=_AMBER)
            _text(combined, f"cam1 {mode_label}", (w + 10, 26), color=_GREEN)
            _text(combined,
                  f"RMS={calib['rms']:.3f}px  baseline={calib['baseline']:.1f}mm  "
                  f"fps cam0={cams.cam0.fps:.1f}  cam1={cams.cam1.fps:.1f}",
                  (10, 54), 0.48, _WHITE)
            _text(combined, "Q=quit  R=raw/rectified  L=lines",
                  (10, h - 12), 0.42, (160, 160, 160))

            disp = cv2.resize(combined, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("ArcheryEdge — live stereo", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                show_rectified = not show_rectified
            if key == ord("l"):
                show_lines = not show_lines

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
