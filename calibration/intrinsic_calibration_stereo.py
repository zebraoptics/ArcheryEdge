"""
calibration/intrinsic_calibration_stereo.py
============================================
Single-camera intrinsic calibration using a ChArUco board.
Captures from Jetson CSI cameras via utils.CSICamera (GStreamer / nvarguscamerasrc).
Calibrate one camera at a time using --camera-id.

Workflow
--------
1. Run in CAPTURE mode to collect frames with live coverage guidance.
   Press SPACE to capture, Q to quit when coverage is sufficient.

2. Run in CALIBRATE mode to process saved frames and output intrinsics.

3. Run in VERIFY mode to visually confirm undistortion on a live feed.

Outputs
-------
  data/intrinsics_<camera_id>.npz
    K      : (3,3) camera matrix
    D      : (5,)  distortion coefficients [k1,k2,p1,p2,k3]
    rms    : scalar reprojection RMS error (target < 0.5 px)
    img_size: (w, h)

Usage examples
--------------
  # Step 1 — capture frames from CSI camera 0
  python calibration/intrinsic_calibration_stereo.py --mode capture --camera-id 0

  # Step 1 — capture frames from CSI camera 1
  python calibration/intrinsic_calibration_stereo.py --mode capture --camera-id 1

  # Step 2 — calibrate from saved frames
  python calibration/intrinsic_calibration_stereo.py --mode calibrate --camera-id 0

  # Step 3 — verify undistortion live
  python calibration/intrinsic_calibration_stereo.py --mode verify --camera-id 0

  # All three steps in one go
  python calibration/intrinsic_calibration_stereo.py --mode all --camera-id 0

ChArUco board parameters (must match your Android app!)
-------------------------------------------------------
  --squares-x   : number of chessboard squares horizontally (default 6)
  --squares-y   : number of chessboard squares vertically   (default 5)
  --square-size : physical size of one chessboard square in metres (default 0.04)
  --marker-size : physical size of ArUco marker in metres   (default 0.03)
  --aruco-dict  : ArUco dictionary name                     (default DICT_4X4_50)

IMX219 sensor modes (--sensor-mode)
------------------------------------
  -1  auto (default)
   0  3280x2464 @ 21fps  — full resolution
   1  3280x1848 @ 28fps  — full, cropped
   2  1920x1080 @ 30fps  — partial 2x2 bin
   3  1640x1232 @ 30fps  — 2x2 binning
   4  1640x922  @ 30fps  — 2x2 binning, cropped
   5  1280x720  @ 60fps  — 2x2 binning
   6  1280x720  @ 120fps — 2x2 binning

Requirements
------------
  pip install opencv-contrib-python numpy
  GStreamer with nvarguscamerasrc (Jetson JetPack)
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# ChArUco board — fixed for this rig
# ─────────────────────────────────────────────
BOARD_SQUARES_X  = 6
BOARD_SQUARES_Y  = 9
BOARD_SQUARE_SIZE = 0.03   # metres (18.4 mm)
BOARD_MARKER_SIZE = 0.03 * 6.0 / 8.0  # metres (16.56 mm)
BOARD_ARUCO_DICT  = "DICT_4X4_100"

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CameraConfig, CSICamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("stereo_cam_calib")

# ─────────────────────────────────────────────
# ChArUco board factory
# ─────────────────────────────────────────────

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_50": aruco.DICT_7X7_50,
}


def make_board(squares_x: int, squares_y: int, square_size: float,
               marker_size: float, dict_name: str) -> tuple:
    """Create a ChArUco board and its ArUco dictionary.

    Returns (board, aruco_dict, detector_params)
    """
    adict = aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size,
        marker_size,
        adict,
    )
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return board, adict, params


# ─────────────────────────────────────────────
# Coverage tracker
# ─────────────────────────────────────────────


@dataclass
class CoverageTracker:
    """
    Tracks pose diversity across six dimensions:
      - Image region  (3×3 grid of frame quadrants)
      - Depth         (near / mid / far)
      - Tilt          (X-axis rotation, 4 bins)
      - Pan           (Y-axis rotation, 5 bins)
      - Roll          (Z-axis rotation, 5 bins)
      - Corners seen  (fraction of board corners detected)
    """
    regions: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=int))
    depth: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=int))
    pan: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=int))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=int))
    n_captured: int = 0
    TARGET_PER_BIN: int = 2
    MIN_FRAMES: int = 20

    @staticmethod
    def _depth_bin(dist: float) -> int:
        if dist < 0.50: return 0
        if dist < 1.00: return 1
        return 2

    @staticmethod
    def _tilt_bin(deg: float) -> int:
        if deg < -15: return 0
        if deg < 0: return 1
        if deg < 15: return 2
        return 3

    @staticmethod
    def _pan_bin(deg: float) -> int:
        if deg < -30: return 0
        if deg < -10: return 1
        if deg < 10: return 2
        if deg < 30: return 3
        return 4

    @staticmethod
    def _roll_bin(deg: float) -> int:
        if deg < -20: return 0
        if deg < -7: return 1
        if deg < 7: return 2
        if deg < 20: return 3
        return 4

    @staticmethod
    def _region_bin(corners: np.ndarray, img_w: int, img_h: int) -> int:
        cx = float(np.mean(corners[:, 0, 0]))
        cy = float(np.mean(corners[:, 0, 1]))
        col = min(int(cx / img_w * 3), 2)
        row = min(int(cy / img_h * 3), 2)
        return row * 3 + col

    @staticmethod
    def _rvec_to_euler(rvec: np.ndarray):
        R, _ = cv2.Rodrigues(rvec)
        tilt = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        pan = float(
            np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        return tilt, pan, roll

    def update(self, rvec: np.ndarray, tvec: np.ndarray,
               charuco_corners: np.ndarray, img_w: int, img_h: int) -> dict:
        dist = float(np.linalg.norm(tvec))
        tilt, pan, roll = self._rvec_to_euler(rvec)

        self.depth[self._depth_bin(dist)] += 1
        self.tilt[self._tilt_bin(tilt)] += 1
        self.pan[self._pan_bin(pan)] += 1
        self.roll[self._roll_bin(roll)] += 1
        self.regions[self._region_bin(charuco_corners, img_w, img_h)] += 1
        self.n_captured += 1

        return dict(dist=dist, tilt=tilt, pan=pan, roll=roll)

    def score(self) -> float:
        T = self.TARGET_PER_BIN
        all_bins = np.concatenate(
            [self.regions, self.depth, self.tilt, self.pan, self.roll])
        return float(np.clip(all_bins, 0, T).sum() / (len(all_bins) * T) * 100)

    def is_sufficient(self) -> bool:
        return self.n_captured >= self.MIN_FRAMES and self.score() >= 75.0

    def next_suggestion(self) -> str:
        region_names = [
            "top-left", "top-center", "top-right",
            "middle-left", "center", "middle-right",
            "bottom-left", "bottom-center", "bottom-right",
        ]
        if any(self.regions == 0):
            idx = int(np.argmin(self.regions))
            return f"Move board to {region_names[idx]} of the frame"
        if self.depth[0] < self.TARGET_PER_BIN:
            return "Hold board CLOSER to camera (< 50 cm)"
        if self.depth[2] < self.TARGET_PER_BIN:
            return "Hold board FARTHER from camera (> 1 m)"
        if self.tilt[0] < self.TARGET_PER_BIN or self.tilt[3] < self.TARGET_PER_BIN:
            return "Tilt board UP or DOWN sharply (+/- 25 deg)"
        if self.pan[0] < self.TARGET_PER_BIN or self.pan[4] < self.TARGET_PER_BIN:
            return "Rotate board LEFT or RIGHT (+/- 35 deg)"
        if self.roll[0] < self.TARGET_PER_BIN or self.roll[4] < self.TARGET_PER_BIN:
            return "Roll board clockwise or counter-clockwise (+/- 25 deg)"
        return "Good diversity — keep adding varied poses"

    def summary_lines(self) -> list[str]:
        return [
            f"Captured : {self.n_captured} / {self.MIN_FRAMES} min",
            f"Coverage : {self.score():.0f}% / 75% min",
            f"Regions  : {int((self.regions > 0).sum())} / 9",
            f"Depth    : {list(self.depth)}  (near/mid/far)",
            f"Tilt     : {list(self.tilt)}",
            f"Pan      : {list(self.pan)}",
            f"Roll     : {list(self.roll)}",
        ]


# ─────────────────────────────────────────────
# HUD drawing helpers
# ─────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 220, 100)
_AMBER = (0, 180, 240)
_RED = (50, 50, 220)
_WHITE = (230, 230, 230)
_DARK = (20, 20, 20)
_BLUE = (220, 120, 30)


def _text(img, txt, xy, scale=0.55, color=_WHITE, thickness=1):
    cv2.putText(img, txt, xy, _FONT, scale, _DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_coverage_hud(img: np.ndarray,
                      tracker: CoverageTracker,
                      last_metrics: Optional[dict] = None) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()

    panel_w = 310
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    score = tracker.score()
    score_color = _GREEN if score >= 75 else (_AMBER if score >= 40 else _RED)
    sufficient = tracker.is_sufficient()

    y = 28
    _text(img, "INTRINSIC CALIBRATION", (10, y), 0.50, _WHITE)
    y += 24

    _text(img, f"Frames: {tracker.n_captured}   Score: {score:.0f}%", (10, y),
          0.52, score_color)
    y += 18
    bar_w = panel_w - 20
    filled = int(bar_w * score / 100)
    cv2.rectangle(img, (10, y), (10 + bar_w, y + 7), (60, 60, 60), -1)
    cv2.rectangle(img, (10, y), (10 + filled, y + 7),
                  (0, 180, 80) if score >= 75 else (0, 160, 200), -1)
    y += 18

    status_txt = "READY — press C to calibrate" if sufficient else "Collecting..."
    _text(img, status_txt, (10, y), 0.48, _GREEN if sufficient else _AMBER)
    y += 22

    _text(img, "Regions (3x3):", (10, y), 0.46, _WHITE)
    y += 16
    cell = 28
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            cnt = tracker.regions[idx]
            x0 = 10 + col * (cell + 3)
            y0 = y + row * (cell + 3)
            clr = (30, 140, 70) if cnt >= tracker.TARGET_PER_BIN \
                  else (30, 120, 160) if cnt > 0 else (60, 60, 60)
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), clr, -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (100, 100, 100), 1)
            _text(img, str(cnt), (x0 + 8, y0 + 19), 0.42, _WHITE)
    y += 3 * (cell + 3) + 8

    depth_labels = ["Near", "Mid ", "Far "]
    _text(img, "Depth:", (10, y), 0.46, _WHITE)
    y += 16
    for i, lbl in enumerate(depth_labels):
        cnt = tracker.depth[i]
        bfill = min(cnt / tracker.TARGET_PER_BIN, 1.0)
        clr = (30, 140, 70) if cnt >= tracker.TARGET_PER_BIN \
              else (30, 120, 160) if cnt > 0 else (60, 60, 60)
        _text(img, lbl, (10, y), 0.42, _WHITE)
        bx = 60
        bw = panel_w - bx - 30
        cv2.rectangle(img, (bx, y - 10), (bx + bw, y), (60, 60, 60), -1)
        cv2.rectangle(img, (bx, y - 10), (bx + int(bw * bfill), y), clr, -1)
        _text(img, str(cnt), (bx + bw + 4, y), 0.40, _WHITE)
        y += 16

    y += 4

    def _rot_row(label, data):
        nonlocal y
        _text(img, f"{label}:", (10, y), 0.42, _WHITE)
        for i, cnt in enumerate(data):
            bx = 55 + i * 48
            clr = (30, 140, 70) if cnt >= tracker.TARGET_PER_BIN \
                  else (30, 120, 160) if cnt > 0 else (60, 60, 60)
            cv2.rectangle(img, (bx, y - 12), (bx + 40, y + 2), clr, -1)
            _text(img, str(cnt), (bx + 13, y), 0.40, _WHITE)
        y += 18

    _rot_row("Tilt", tracker.tilt)
    _rot_row("Pan ", tracker.pan[:4])
    _rot_row("Roll", tracker.roll[:4])
    y += 4

    suggestion = tracker.next_suggestion()
    words = suggestion.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 36:
            lines.append(cur)
            cur = w
        else:
            cur += (" " if cur else "") + w
    if cur:
        lines.append(cur)

    _text(img, "Next pose:", (10, y), 0.46, _AMBER)
    y += 18
    for line in lines[:3]:
        _text(img, line, (10, y), 0.44, _WHITE)
        y += 17

    if last_metrics:
        y += 6
        _text(img,
              f"dist={last_metrics['dist']:.2f}m  "
              f"tilt={last_metrics['tilt']:.0f}  "
              f"pan={last_metrics['pan']:.0f}  "
              f"roll={last_metrics['roll']:.0f}",
              (10, y), 0.40, (180, 180, 180))

    _text(img, "SPACE=capture  C=calibrate  Q=quit", (10, h - 12), 0.40,
          (150, 150, 150))

    return img


def draw_detection_overlay(img, charuco_corners, charuco_ids, marker_corners,
                           board, K, D, rvec=None, tvec=None):
    if marker_corners:
        aruco.drawDetectedMarkers(img, marker_corners)
    if charuco_ids is not None and len(charuco_ids) >= 4:
        aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids,
                                         (0, 255, 120))
    if rvec is not None and tvec is not None and K is not None:
        axis_len = board.getSquareLength() * 3
        cv2.drawFrameAxes(img, K, D, rvec, tvec, axis_len, 2)
    return img


# ─────────────────────────────────────────────
# Calibration data storage
# ─────────────────────────────────────────────


@dataclass
class CalibFrame:
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    img_size: tuple
    frame_path: Path
    metrics: dict


# ─────────────────────────────────────────────
# STEP 1 — Capture (CSI camera)
# ─────────────────────────────────────────────


def _make_csi_config(args) -> CameraConfig:
    return CameraConfig(
        sensor_id=args.camera_id,
        width=args.width,
        height=args.height,
        framerate=args.fps,
        flip_method=args.flip_method,
        sensor_mode=args.sensor_mode,
    )


def run_capture(args) -> None:
    """
    Live capture session with coverage HUD using CSI camera.

    Controls
    --------
    SPACE  : capture current frame (only if detection is valid)
    C      : trigger calibration immediately (if sufficient)
    Q      : quit
    """
    out_dir = Path(args.output) / f"cam{args.camera_id}" / "raw_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    board, adict, params = make_board(
        args.squares_x, args.squares_y,
        args.square_size, args.marker_size, args.aruco_dict,
    )
    charuco_detector = aruco.CharucoDetector(board)
    tracker = CoverageTracker(MIN_FRAMES=args.min_frames)

    cfg = _make_csi_config(args)
    log.info("Opening CSI camera %d — %dx%d @ %dfps  sensor_mode=%d  flip=%d",
             cfg.sensor_id, cfg.width, cfg.height, cfg.framerate,
             cfg.sensor_mode, cfg.flip_method)

    with CSICamera(cfg) as cam:
        # Warm up — wait until first frame arrives
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cam.read() is not None:
                break
            time.sleep(0.05)
        else:
            log.error("Camera %d did not produce a frame within 5 s", args.camera_id)
            sys.exit(1)

        log.info("Capture started — output: %s", out_dir)
        last_metrics = None
        last_charuco_corners = None
        last_charuco_ids = None
        last_marker_corners = None
        K_approx = None
        D_approx = np.zeros(5)

        saved_count = 0

        while True:
            raw = cam.read()
            if raw is None:
                continue

            h, w = raw.shape[:2]
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            disp = raw.copy()

            charuco_corners, charuco_ids, marker_corners, _ = \
                charuco_detector.detectBoard(gray)

            valid_detection = (charuco_ids is not None
                               and len(charuco_ids) >= args.min_corners)

            rvec, tvec = None, None
            if valid_detection and K_approx is not None:
                obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                if obj_pts is not None and len(obj_pts) >= 4:
                    ok_pose, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K_approx, D_approx
                    )
                    if ok_pose:
                        last_charuco_corners = charuco_corners
                        last_charuco_ids = charuco_ids
                        last_marker_corners = marker_corners
                    else:
                        rvec, tvec = None, None

            if K_approx is None:
                f = max(w, h)
                K_approx = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                                    dtype=np.float64)

            disp = draw_detection_overlay(
                disp, charuco_corners, charuco_ids, marker_corners,
                board, K_approx, D_approx, rvec, tvec,
            )

            if valid_detection:
                n_det = len(charuco_ids)
                txt = f"Detected: {n_det} corners"
                color = _GREEN if n_det >= args.min_corners * 2 else _AMBER
            else:
                txt = f"No board (need >= {args.min_corners} corners)"
                color = _RED
            _text(disp, txt, (w - 280, 28), 0.52, color)

            # FPS overlay (top-right)
            _text(disp, f"cam{args.camera_id}  {cam.fps:.1f} fps",
                  (w - 280, 56), 0.52, _WHITE)

            disp = draw_coverage_hud(disp, tracker, last_metrics)

            if args.display_scale != 1.0:
                disp = cv2.resize(disp, None, fx=args.display_scale,
                                  fy=args.display_scale,
                                  interpolation=cv2.INTER_AREA)
            if args.rotate == 90:
                disp = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                disp = cv2.rotate(disp, cv2.ROTATE_180)
            elif args.rotate == 270:
                disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(f"Intrinsic calibration — CSI cam {args.camera_id}", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not valid_detection:
                    log.warning("No valid ChArUco detection — skipped")
                elif rvec is None:
                    log.warning("Pose estimation failed — move board and retry")
                else:
                    metrics = tracker.update(rvec, tvec, charuco_corners, w, h)
                    last_metrics = metrics
                    saved_count += 1
                    fname = out_dir / f"frame_{saved_count:04d}.png"
                    cv2.imwrite(str(fname), raw)
                    log.info(
                        "Captured #%d → %s | dist=%.2fm tilt=%.0f pan=%.0f roll=%.0f",
                        saved_count, fname.name,
                        metrics["dist"], metrics["tilt"],
                        metrics["pan"], metrics["roll"],
                    )
                    for line in tracker.summary_lines():
                        log.info("  %s", line)

            elif key == ord("c"):
                if tracker.n_captured < 5:
                    log.warning("Need at least 5 frames — only %d captured",
                                tracker.n_captured)
                else:
                    log.info("Triggering calibration from capture mode…")
                    cv2.destroyAllWindows()
                    _run_calibration_from_dir(out_dir, board, charuco_detector, args)
                    return

            elif key == ord("q"):
                break

    cv2.destroyAllWindows()
    log.info("Capture ended — %d frames saved to %s", saved_count, out_dir)
    if tracker.is_sufficient():
        log.info("Coverage sufficient — run with --mode calibrate to process")
    else:
        log.warning("Coverage score %.0f%% < 75%% — consider capturing more poses",
                    tracker.score())


# ─────────────────────────────────────────────
# STEP 2 — Calibrate
# ─────────────────────────────────────────────


def _run_calibration_from_dir(frame_dir: Path, board, charuco_detector,
                              args) -> Optional[Path]:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        log.error("No PNG frames found in %s", frame_dir)
        return None

    log.info("Processing %d frames from %s", len(frame_paths), frame_dir)

    all_obj_pts = []
    all_img_pts = []
    img_size = None
    good_count = 0

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < args.min_corners:
            log.warning("  %s — skipped (only %d corners)", fp.name,
                        len(charuco_ids) if charuco_ids is not None else 0)
            continue

        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < args.min_corners:
            log.warning("  %s — matchImagePoints failed, skipped", fp.name)
            continue

        all_obj_pts.append(obj_pts)
        all_img_pts.append(img_pts)
        good_count += 1
        log.info("  %s — %d corners OK", fp.name, len(charuco_ids))

    if good_count < 10:
        log.error("Only %d usable frames (need >= 10). Capture more varied poses.",
                  good_count)
        return None

    log.info("Running cv2.calibrateCamera on %d frames…", good_count)

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_obj_pts,
        imagePoints=all_img_pts,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_RATIONAL_MODEL if args.rational_model else 0,
    )

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Calibration complete")
    log.info("  RMS reprojection error : %.4f px  (target < 0.5)", rms)
    log.info("  Focal length  fx=%.1f  fy=%.1f", K[0, 0], K[1, 1])
    log.info("  Principal pt  cx=%.1f  cy=%.1f", K[0, 2], K[1, 2])
    log.info("  Distortion    %s", np.round(D.ravel(), 5))
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if rms > 1.0:
        log.warning("RMS > 1.0 px — calibration is poor. Check for:")
        log.warning("  • Blurry frames (hold board still when capturing)")
        log.warning("  • Board flexing (tablet screen flat?)")
        log.warning("  • Wrong square_size / marker_size parameters")
        log.warning("  • Insufficient pose diversity")
    elif rms > 0.5:
        log.warning("RMS %.4f px is acceptable but not ideal (target < 0.5)", rms)
    else:
        log.info("Excellent RMS — intrinsics are reliable")

    per_frame_errors = []
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(all_obj_pts[i], rvec, tvec, K, D)
        err = float(np.sqrt(np.mean((proj - all_img_pts[i])**2)))
        per_frame_errors.append((i, err))

    if per_frame_errors:
        errors = [e for _, e in per_frame_errors]
        log.info("Per-frame reprojection error — "
                 "min=%.3f  mean=%.3f  max=%.3f  std=%.3f",
                 min(errors), np.mean(errors), max(errors), np.std(errors))
        bad = [(i, e) for i, e in per_frame_errors if e > rms * 2.5]
        if bad:
            log.warning("High-error frames (consider removing and recalibrating):")
            for i, e in bad:
                log.warning("  frame_%04d.png  error=%.3f px", i + 1, e)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"intrinsics_cam{args.camera_id}.npz"

    np.savez(
        str(npz_path),
        K=K, D=D,
        rms=np.array([rms]),
        img_size=np.array(img_size),
        camera_id=np.array([args.camera_id]),
        n_frames=np.array([good_count]),
        board_squares_x=np.array([args.squares_x]),
        board_squares_y=np.array([args.squares_y]),
        square_size=np.array([args.square_size]),
        marker_size=np.array([args.marker_size]),
    )
    log.info("Intrinsics saved → %s", npz_path)

    json_path = out_dir / f"intrinsics_cam{args.camera_id}.json"
    with open(json_path, "w") as f:
        json.dump({
            "camera_id": args.camera_id,
            "rms_px": round(rms, 6),
            "img_size": list(img_size),
            "K": K.tolist(),
            "D": D.ravel().tolist(),
            "n_frames": good_count,
            "board": {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_size": args.square_size,
                "marker_size": args.marker_size,
                "aruco_dict": args.aruco_dict,
            },
        }, f, indent=2)
    log.info("Human-readable JSON → %s", json_path)

    return npz_path


def run_calibrate(args) -> None:
    frame_dir = Path(args.output) / f"cam{args.camera_id}" / "raw_frames"
    board, _, _ = make_board(
        args.squares_x, args.squares_y,
        args.square_size, args.marker_size, args.aruco_dict,
    )
    charuco_detector = aruco.CharucoDetector(board)
    _run_calibration_from_dir(frame_dir, board, charuco_detector, args)


# ─────────────────────────────────────────────
# STEP 3 — Verify (live undistortion preview)
# ─────────────────────────────────────────────


def run_verify(args) -> None:
    """Load saved intrinsics and show live undistorted feed side-by-side."""
    npz_path = Path(args.output) / f"intrinsics_cam{args.camera_id}.npz"
    if not npz_path.exists():
        log.error("Intrinsics not found at %s — run --mode calibrate first", npz_path)
        sys.exit(1)

    data = np.load(str(npz_path))
    K = data["K"]
    D = data["D"]
    img_size = tuple(data["img_size"])
    rms = float(data["rms"][0])
    log.info("Loaded intrinsics — RMS=%.4f px  K=\n%s", rms, K)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha=0,
                                              newImgSize=img_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, img_size,
                                             cv2.CV_16SC2)

    cfg = _make_csi_config(args)
    snap_count = 0
    snap_dir = Path(args.output) / f"cam{args.camera_id}" / "verify_snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)

    log.info("Verify mode — Q to quit, S to save snapshot")

    with CSICamera(cfg) as cam:
        # Warm up
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cam.read() is not None:
                break
            time.sleep(0.05)
        else:
            log.error("Camera %d did not produce a frame within 5 s", args.camera_id)
            sys.exit(1)

        while True:
            frame = cam.read()
            if frame is None:
                continue

            undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            h, w = frame.shape[:2]
            combined = np.hstack([frame, undist])
            _text(combined, "Original", (10, 28), 0.7, _AMBER)
            _text(combined, "Undistorted", (w + 10, 28), 0.7, _GREEN)
            _text(combined, f"RMS={rms:.4f}px", (w + 10, 56), 0.55, _WHITE)
            _text(combined, f"cam{args.camera_id}  {cam.fps:.1f} fps",
                  (10, 56), 0.52, _WHITE)
            _text(combined, "Q=quit  S=snapshot", (10, h - 12), 0.45,
                  (150, 150, 150))

            disp_combined = combined
            if args.display_scale != 1.0:
                disp_combined = cv2.resize(combined, None, fx=args.display_scale,
                                           fy=args.display_scale,
                                           interpolation=cv2.INTER_AREA)
            if args.rotate == 90:
                disp_combined = cv2.rotate(disp_combined, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                disp_combined = cv2.rotate(disp_combined, cv2.ROTATE_180)
            elif args.rotate == 270:
                disp_combined = cv2.rotate(disp_combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(f"Verify intrinsics — CSI cam {args.camera_id}", disp_combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap_count += 1
                ts = f"{time.monotonic():.2f}".replace(".", "_")
                cv2.imwrite(str(snap_dir / f"orig_{snap_count:03d}_{ts}.png"), frame)
                cv2.imwrite(str(snap_dir / f"undist_{snap_count:03d}_{ts}.png"), undist)
                log.info("Snapshot %d saved", snap_count)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single CSI camera intrinsic calibration (ChArUco)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode",
                   choices=["capture", "calibrate", "verify", "all"],
                   default="all",
                   help="capture=collect frames | calibrate=compute K,D | "
                        "verify=live undistortion | all=run all three")

    # Camera identity — maps directly to CSI sensor_id
    p.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="CSI sensor_id of the camera to calibrate (0=left, 1=right). "
             "Used for both the GStreamer pipeline and output filenames.")

    # Output
    p.add_argument("--output", type=str, default="data",
                   help="Root output directory (data/intrinsics_cam0.npz etc.)")

    # CSI / IMX219 camera settings
    p.add_argument("--width", type=int, default=1640,
                   help="Capture width in pixels")
    p.add_argument("--height", type=int, default=1232,
                   help="Capture height in pixels")
    p.add_argument("--fps", type=int, default=30,
                   help="Capture framerate")
    p.add_argument(
        "--sensor-mode", type=int, default=3,
        help="IMX219 sensor mode (-1=auto, 3=1640x1232 2x2bin, "
             "5=1280x720@60, 6=1280x720@120). Must match width/height/fps.")
    p.add_argument(
        "--flip-method", type=int, default=0,
        help="GStreamer nvvidconv flip-method "
             "(0=none, 1=ccw90, 2=rot180, 3=cw90, 4=horiz, 6=vert)")

    # ChArUco board — hardcoded to rig constants, not user-configurable

    # Calibration options
    p.add_argument("--min-corners", type=int, default=6,
                   help="Minimum ChArUco corners required per frame")
    p.add_argument("--min-frames", type=int, default=20,
                   help="Minimum frames for coverage to be 'sufficient'")
    p.add_argument("--rational-model", action="store_true",
                   help="Use rational distortion model (8 coefficients) instead of "
                        "standard (5). Better for wide-angle lenses.")
    p.add_argument("--display-scale", type=float, default=0.5,
                   help="Scale factor for display window (1.0=full size, 0.5=half)")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate display image clockwise by degrees (display only)")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    # Inject hardcoded board constants so downstream functions have them on args
    args.squares_x   = BOARD_SQUARES_X
    args.squares_y   = BOARD_SQUARES_Y
    args.square_size = BOARD_SQUARE_SIZE
    args.marker_size = BOARD_MARKER_SIZE
    args.aruco_dict  = BOARD_ARUCO_DICT
    return args


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("Mode=%s  camera_id=%d  sensor_mode=%d  flip=%d",
             args.mode, args.camera_id, args.sensor_mode, args.flip_method)
    log.info("Resolution: %dx%d @ %dfps", args.width, args.height, args.fps)
    log.info("Board: %dx%d squares  square=%.1fmm  marker=%.2fmm  dict=%s",
             BOARD_SQUARES_X, BOARD_SQUARES_Y,
             BOARD_SQUARE_SIZE * 1000, BOARD_MARKER_SIZE * 1000,
             BOARD_ARUCO_DICT)

    if args.mode in ("capture", "all"):
        run_capture(args)
    if args.mode in ("calibrate", "all"):
        run_calibrate(args)
    if args.mode in ("verify", "all"):
        run_verify(args)


if __name__ == "__main__":
    main()
