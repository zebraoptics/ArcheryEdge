"""
calibration/stereo_calibrate.py
================================
Stereo calibration using a ChArUco board and pre-computed per-camera intrinsics.

Workflow
--------
1. CAPTURE — open both CSI cameras simultaneously and save synchronized pairs.
   Both cameras must see the same board position in each captured frame pair.
   Press SPACE to capture, Q to quit.

2. CALIBRATE — load intrinsics_cam0.npz / intrinsics_cam1.npz, detect ChArUco
   corners in every synchronized pair, and run cv2.stereoCalibrate() with
   CALIB_FIX_INTRINSIC to solve only for the inter-camera R and T.

3. VERIFY — live side-by-side display of epipolar-rectified frames so you can
   confirm horizontal alignment of corresponding points.

Outputs
-------
  data/stereo_calib.npz
    K0, D0, K1, D1   : per-camera intrinsics (copied from .npz files)
    R, T              : rotation/translation cam1 relative to cam0
    E, F              : essential and fundamental matrices
    R0, R1            : per-camera rectification rotations
    P0, P1            : projection matrices in rectified coordinates
    Q                 : disparity-to-depth mapping matrix
    map0x, map0y      : undistort+rectify remap for cam0
    map1x, map1y      : undistort+rectify remap for cam1
    img_size          : (w, h) used during calibration
    rms               : stereo calibration RMS reprojection error

  data/stereo_calib.json  — human-readable subset of the above

Usage examples
--------------
  # Step 1 — capture synchronized stereo pairs
  python calibration/stereo_calibrate.py --mode capture

  # Step 2 — compute stereo calibration from saved pairs
  python calibration/stereo_calibrate.py --mode calibrate

  # Step 3 — verify rectification live
  python calibration/stereo_calibrate.py --mode verify

  # All three steps back-to-back
  python calibration/stereo_calibrate.py --mode all

ChArUco board parameters (must match intrinsic_calibration_stereo.py!)
-----------------------------------------------------------------------
  6 x 9 squares, 30 mm square, DICT_4X4_100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CameraConfig, DualCameraConfig, DualCSICamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("stereo_calib")

# ── Board constants — must match intrinsic_calibration_stereo.py ──────────────
BOARD_SQUARES_X   = 6
BOARD_SQUARES_Y   = 9
BOARD_SQUARE_SIZE = 0.03          # metres
BOARD_MARKER_SIZE = 0.03 * 6 / 8 # metres
BOARD_ARUCO_DICT  = "DICT_4X4_100"

ARUCO_DICTS = {
    "DICT_4X4_50":  aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_50":  aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_50":  aruco.DICT_6X6_50,
    "DICT_6X6_250": aruco.DICT_6X6_250,
}

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 220, 100)
_AMBER = (0, 180, 240)
_RED   = (50,  50, 220)
_WHITE = (230, 230, 230)


def _text(img, txt, xy, scale=0.55, color=_WHITE, thickness=1):
    cv2.putText(img, txt, xy, _FONT, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, _FONT, scale, color,       thickness,     cv2.LINE_AA)


def _make_board():
    adict  = aruco.getPredefinedDictionary(ARUCO_DICTS[BOARD_ARUCO_DICT])
    board  = aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        BOARD_SQUARE_SIZE, BOARD_MARKER_SIZE, adict,
    )
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return board, aruco.CharucoDetector(board)


def _load_intrinsics(npz_path: Path):
    if not npz_path.exists():
        log.error("Intrinsics file not found: %s", npz_path)
        sys.exit(1)
    d = np.load(str(npz_path))
    log.info("Loaded %s  RMS=%.4f px", npz_path.name, float(d["rms"][0]))
    return d["K"].copy(), d["D"].copy(), tuple(d["img_size"].tolist())


# Per-camera mounting rotation (clockwise degrees needed to make the raw
# sensor frame upright). Must match intrinsic_calibration_stereo.py so that
# K/D/R/T all live in the same upright coordinate frame.
_CAM0_ROTATION = 270
_CAM1_ROTATION = 90


def _dual_cfg(args) -> DualCameraConfig:
    cam_cfg = dict(
        width=args.width, height=args.height,
        framerate=args.fps, flip_method=args.flip_method,
        sensor_mode=args.sensor_mode,
    )
    cfg        = DualCameraConfig()
    cfg.cam0   = CameraConfig(sensor_id=0, rotation=_CAM0_ROTATION, **cam_cfg)
    cfg.cam1   = CameraConfig(sensor_id=1, rotation=_CAM1_ROTATION, **cam_cfg)
    return cfg


# ── STEP 1: CAPTURE ───────────────────────────────────────────────────────────

def run_capture(args) -> None:
    out0 = Path(args.output) / "stereo_pairs" / "cam0"
    out1 = Path(args.output) / "stereo_pairs" / "cam1"
    out0.mkdir(parents=True, exist_ok=True)
    out1.mkdir(parents=True, exist_ok=True)

    _, detector = _make_board()
    saved = 0

    log.info("Opening both CSI cameras …")
    with DualCSICamera(_dual_cfg(args)) as cams:
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

        log.info("Ready — SPACE=capture pair  Q=quit  (need ~20+ pairs)")

        while True:
            try:
                f0, f1 = cams.read_sync(timeout=1.0, max_dt=0.015)
            except TimeoutError:
                log.warning("Frame sync timeout — retrying")
                continue

            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

            corners0, ids0, _, _ = detector.detectBoard(g0)
            corners1, ids1, _, _ = detector.detectBoard(g1)

            det0 = ids0 is not None and len(ids0) >= args.min_corners
            det1 = ids1 is not None and len(ids1) >= args.min_corners

            disp0 = f0.copy()
            disp1 = f1.copy()
            if det0:
                aruco.drawDetectedCornersCharuco(disp0, corners0, ids0, (0, 255, 120))
            if det1:
                aruco.drawDetectedCornersCharuco(disp1, corners1, ids1, (0, 255, 120))

            both_ok = det0 and det1
            status  = f"Pairs: {saved}  Board: {'BOTH OK' if both_ok else 'waiting…'}"
            color   = _GREEN if both_ok else _AMBER
            _text(disp0, status, (10, 28), color=color)
            _text(disp1, f"cam1  {'OK' if det1 else 'NO BOARD'}", (10, 28),
                  color=_GREEN if det1 else _RED)
            _text(disp0, "SPACE=capture  Q=quit", (10, disp0.shape[0] - 12),
                  0.45, _WHITE)

            scale = args.display_scale
            combined = np.hstack([
                cv2.resize(disp0, None, fx=scale, fy=scale),
                cv2.resize(disp1, None, fx=scale, fy=scale),
            ])
            cv2.imshow("Stereo capture — cam0 | cam1", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                if not both_ok:
                    log.warning("Board not visible in both cameras — skipped")
                    continue
                saved += 1
                fname = f"pair_{saved:04d}.png"
                cv2.imwrite(str(out0 / fname), f0)
                cv2.imwrite(str(out1 / fname), f1)
                log.info("Saved pair %d  cam0:%d corners  cam1:%d corners",
                         saved, len(ids0), len(ids1))

    cv2.destroyAllWindows()
    log.info("Capture done — %d synchronized pairs saved", saved)
    if saved < 10:
        log.warning("Only %d pairs — recommend >= 20 for reliable stereo calibration",
                    saved)


# ── STEP 2: CALIBRATE ─────────────────────────────────────────────────────────

def run_calibrate(args) -> None:
    data_dir = Path(args.output)
    K0, D0, img_size = _load_intrinsics(data_dir / "intrinsics_cam0.npz")
    K1, D1, _        = _load_intrinsics(data_dir / "intrinsics_cam1.npz")

    board, detector = _make_board()

    pair_dir0 = data_dir / "stereo_pairs" / "cam0"
    pair_dir1 = data_dir / "stereo_pairs" / "cam1"
    frames0 = sorted(pair_dir0.glob("*.png"))
    frames1 = sorted(pair_dir1.glob("*.png"))

    if not frames0:
        log.error("No stereo pairs found in %s — run --mode capture first", pair_dir0)
        sys.exit(1)
    if len(frames0) != len(frames1):
        log.error("Pair mismatch: %d cam0 vs %d cam1 images", len(frames0), len(frames1))
        sys.exit(1)

    log.info("Processing %d stereo pairs …", len(frames0))

    obj_pts_list = []
    img_pts0_list = []
    img_pts1_list = []
    good = 0

    for p0, p1 in zip(frames0, frames1):
        img0 = cv2.imread(str(p0))
        img1 = cv2.imread(str(p1))
        g0   = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        g1   = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        cc0, ids0, _, _ = detector.detectBoard(g0)
        cc1, ids1, _, _ = detector.detectBoard(g1)

        if ids0 is None or ids1 is None:
            log.warning("  %s — board not detected in one or both images, skipped",
                        p0.name)
            continue

        # Find common corner IDs
        common = np.intersect1d(ids0.ravel(), ids1.ravel())
        if len(common) < args.min_corners:
            log.warning("  %s — only %d common corners (need %d), skipped",
                        p0.name, len(common), args.min_corners)
            continue

        mask0 = np.isin(ids0.ravel(), common)
        mask1 = np.isin(ids1.ravel(), common)
        cc0_filt = cc0[mask0]
        cc1_filt = cc1[mask1]
        ids_filt = ids0[mask0]

        obj_pts, _ = board.matchImagePoints(cc0_filt, ids_filt)
        if obj_pts is None or len(obj_pts) < args.min_corners:
            log.warning("  %s — matchImagePoints failed, skipped", p0.name)
            continue

        # img_pts must align with obj_pts using same filtered corners
        _, img_pts0 = board.matchImagePoints(cc0_filt, ids_filt)
        _, img_pts1 = board.matchImagePoints(cc1_filt, ids1[mask1])
        if img_pts0 is None or img_pts1 is None:
            continue
        if len(img_pts0) != len(img_pts1):
            log.warning("  %s — corner count mismatch after filtering, skipped", p0.name)
            continue

        obj_pts_list.append(obj_pts)
        img_pts0_list.append(img_pts0)
        img_pts1_list.append(img_pts1)
        good += 1
        log.info("  %s — %d common corners OK", p0.name, len(common))

    if good < 8:
        log.error("Only %d usable pairs (need >= 8). Capture more synchronized pairs.",
                  good)
        sys.exit(1)

    log.info("Running cv2.stereoCalibrate on %d pairs …", good)
    flags = cv2.CALIB_FIX_INTRINSIC  # intrinsics already computed, solve only R,T

    rms, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=obj_pts_list,
        imagePoints1=img_pts0_list,
        imagePoints2=img_pts1_list,
        cameraMatrix1=K0, distCoeffs1=D0,
        cameraMatrix2=K1, distCoeffs2=D1,
        imageSize=img_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Stereo calibration complete")
    log.info("  RMS reprojection error : %.4f px  (target < 1.0)", rms)
    baseline = float(np.linalg.norm(T))
    log.info("  Baseline               : %.1f mm", baseline * 1000)
    log.info("  T (translation)        : %s m", np.round(T.ravel(), 4))
    log.info("  R (rodrigues)          : %s deg",
             np.round(np.degrees(cv2.Rodrigues(R)[0].ravel()), 2))
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if rms > 2.0:
        log.warning("High RMS — check that board was truly stationary during each pair capture")
    elif rms > 1.0:
        log.warning("RMS > 1.0 px — acceptable but not ideal; more varied pairs may help")

    # Stereo rectification
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K0, distCoeffs1=D0,
        cameraMatrix2=K1, distCoeffs2=D1,
        imageSize=img_size,
        R=R, T=T,
        alpha=0,           # crop to valid pixels only
        newImageSize=img_size,
    )

    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, img_size, cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)

    out_npz = data_dir / "stereo_calib.npz"
    np.savez(
        str(out_npz),
        K0=K0, D0=D0, K1=K1, D1=D1,
        R=R, T=T, E=E, F=F,
        R0=R0, R1=R1, P0=P0, P1=P1, Q=Q,
        map0x=map0x, map0y=map0y,
        map1x=map1x, map1y=map1y,
        img_size=np.array(img_size),
        rms=np.array([rms]),
        n_pairs=np.array([good]),
    )
    log.info("Stereo calibration saved → %s", out_npz)

    out_json = data_dir / "stereo_calib.json"
    with open(out_json, "w") as f:
        json.dump({
            "rms_px":    round(rms, 6),
            "baseline_m": round(baseline, 6),
            "img_size":  list(img_size),
            "n_pairs":   good,
            "K0": K0.tolist(), "D0": D0.ravel().tolist(),
            "K1": K1.tolist(), "D1": D1.ravel().tolist(),
            "R":  R.tolist(),  "T":  T.ravel().tolist(),
            "R0": R0.tolist(), "P0": P0.tolist(),
            "R1": R1.tolist(), "P1": P1.tolist(),
            "Q":  Q.tolist(),
        }, f, indent=2)
    log.info("Human-readable JSON → %s", out_json)


# ── STEP 3: VERIFY ────────────────────────────────────────────────────────────

def run_verify(args) -> None:
    npz_path = Path(args.output) / "stereo_calib.npz"
    if not npz_path.exists():
        log.error("Stereo calibration not found: %s — run --mode calibrate first",
                  npz_path)
        sys.exit(1)

    d        = np.load(str(npz_path))
    map0x    = d["map0x"]; map0y = d["map0y"]
    map1x    = d["map1x"]; map1y = d["map1y"]
    rms      = float(d["rms"][0])
    baseline = float(np.linalg.norm(d["T"])) * 1000
    log.info("Loaded stereo_calib.npz  RMS=%.4f px  baseline=%.1f mm", rms, baseline)

    log.info("Opening both CSI cameras …")
    with DualCSICamera(_dual_cfg(args)) as cams:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            f0, f1 = cams.read()
            if f0 is not None and f1 is not None:
                break
            time.sleep(0.05)
        else:
            log.error("Cameras did not produce frames within 8 s")
            sys.exit(1)

        log.info("Verify mode — Q to quit")
        while True:
            try:
                f0, f1 = cams.read_sync(timeout=1.0, max_dt=0.015)
            except TimeoutError:
                continue

            rect0 = cv2.remap(f0, map0x, map0y, cv2.INTER_LINEAR)
            rect1 = cv2.remap(f1, map1x, map1y, cv2.INTER_LINEAR)

            h, w = rect0.shape[:2]
            combined = np.hstack([rect0, rect1])

            # Epipolar lines are horizontal in rectified coordinates.
            for y in range(0, h, 60):
                cv2.line(combined, (0, y), (w * 2, y), (0, 200, 200), 1)

            _text(combined, "cam0 rectified", (10, 26), color=_AMBER)
            _text(combined, "cam1 rectified", (w + 10, 26), color=_GREEN)
            _text(combined, f"RMS={rms:.3f}px  baseline={baseline:.1f}mm",
                  (10, 54), 0.48, _WHITE)
            _text(combined, "Epipolar lines should pass through matching features",
                  (10, h - 12), 0.42, (160, 160, 160))

            scale = args.display_scale
            disp  = cv2.resize(combined, None, fx=scale, fy=scale)
            cv2.imshow("Stereo rectification verify", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stereo calibration — ChArUco board, fixed intrinsics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["capture", "calibrate", "verify", "all"],
                   default="all",
                   help="capture=collect pairs | calibrate=compute R,T | "
                        "verify=live rectification | all=run all three")
    p.add_argument("--output", default="data",
                   help="Root data directory (contains intrinsics_cam*.npz)")

    p.add_argument("--width",       type=int,   default=1640)
    p.add_argument("--height",      type=int,   default=1232)
    p.add_argument("--fps",         type=int,   default=30)
    p.add_argument("--sensor-mode", type=int,   default=3,
                   help="IMX219 sensor mode (3=1640x1232 2x2bin)")
    p.add_argument("--flip-method", type=int,   default=0)
    p.add_argument("--display-scale", type=float, default=0.4,
                   help="Scale for preview windows (combined side-by-side is wide)")
    p.add_argument("--min-corners", type=int,   default=6,
                   help="Minimum shared ChArUco corners per stereo pair")
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("Mode=%s  resolution=%dx%d  sensor_mode=%d  flip=%d",
             args.mode, args.width, args.height, args.sensor_mode, args.flip_method)
    log.info("Board: %dx%d squares  square=%.1fmm  marker=%.2fmm  dict=%s",
             BOARD_SQUARES_X, BOARD_SQUARES_Y,
             BOARD_SQUARE_SIZE * 1000, BOARD_MARKER_SIZE * 1000, BOARD_ARUCO_DICT)

    if args.mode in ("capture", "all"):
        run_capture(args)
    if args.mode in ("calibrate", "all"):
        run_calibrate(args)
    if args.mode in ("verify", "all"):
        run_verify(args)


if __name__ == "__main__":
    main()
