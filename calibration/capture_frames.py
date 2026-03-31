"""
capture_frames.py
=================
Stereo frame capture for the archery 3D pose system.

Backends
--------
  mac       Two USB/built-in webcams via OpenCV VideoCapture (default)
  jetson    Two CSI cameras via GStreamer pipelines on Jetson Orin Nano Super

Usage
-----
  # Mac — preview both cameras live (press Q to quit)
  python capture_frames.py --backend mac --preview

  # Mac — capture a calibration session (saves chessboard pairs)
  python capture_frames.py --backend mac --mode calibration --output ./calib_frames

  # Mac — capture a shooting session
  python capture_frames.py --backend mac --mode session --output ./session_frames

  # Jetson (swap backend when hardware arrives)
  python capture_frames.py --backend jetson --preview

Requirements
------------
  pip install opencv-python numpy
"""

import abc
import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("capture_frames")


# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────

@dataclass
class FramePair:
    """A synchronised pair of frames from left and right cameras."""
    left: np.ndarray
    right: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def both_valid(self) -> bool:
        return self.left is not None and self.right is not None


@dataclass
class CameraConfig:
    """Shared configuration for both cameras."""
    width: int = 1280
    height: int = 720
    fps: int = 30


# ─────────────────────────────────────────────
# Abstract base — swap backends here
# ─────────────────────────────────────────────

class StereoCaptureBase(abc.ABC):
    """
    Abstract stereo camera interface.

    Subclass this to add new backends (Jetson CSI, RealSense, etc.).
    All subclasses must implement open(), read(), close(), and
    optionally the is_synced property.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._opened = False

    @abc.abstractmethod
    def open(self) -> bool:
        """Open both cameras. Returns True on success."""

    @abc.abstractmethod
    def read(self) -> FramePair:
        """Read one synchronised frame pair. Called every frame."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all camera resources."""

    @property
    def is_synced(self) -> bool:
        """
        True if the backend provides hardware-level frame synchronisation.
        Mac webcams are software-synced (small temporal offset).
        Jetson with GPIO trigger is hardware-synced.
        """
        return False

    def __enter__(self):
        if not self.open():
            raise RuntimeError(f"{self.__class__.__name__}: failed to open cameras")
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────
# Backend 1 — Mac / USB webcams
# ─────────────────────────────────────────────

class MacStereoCapture(StereoCaptureBase):
    """
    Two webcams via OpenCV VideoCapture.

    On a MacBook with one built-in camera plus one USB camera:
      camera_index_left=0   → built-in FaceTime HD (treat as "left")
      camera_index_right=1  → USB webcam (treat as "right")

    For two USB cameras, try indices 0,1 or 1,2 depending on
    the order macOS enumerates them.

    Sync note
    ---------
    Software sync only — reads are sequential, not simultaneous.
    Typical inter-frame gap is 5–15 ms at 30 fps, which introduces
    a small stereo baseline temporal error. Acceptable for development
    and slow movements; replace with Jetson hardware sync for production.
    """

    def __init__(
        self,
        config: CameraConfig,
        camera_index_left: int = 0,
        camera_index_right: int = 1,
    ):
        super().__init__(config)
        self._idx_l = camera_index_left
        self._idx_r = camera_index_right
        self._cap_l: Optional[cv2.VideoCapture] = None
        self._cap_r: Optional[cv2.VideoCapture] = None

    def _configure_cap(self, cap: cv2.VideoCapture, label: str) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Read back actual values — webcams may not honour exact requests
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("%s camera: %dx%d @ %.1f fps", label, w, h, fps)

    def open(self) -> bool:
        log.info("Opening Mac webcams (L=%d, R=%d)", self._idx_l, self._idx_r)
        self._cap_l = cv2.VideoCapture(self._idx_l)
        self._cap_r = cv2.VideoCapture(self._idx_r)

        if not self._cap_l.isOpened():
            log.error("Could not open left camera (index %d)", self._idx_l)
            return False
        if not self._cap_r.isOpened():
            log.error(
                "Could not open right camera (index %d). "
                "Is a second webcam connected?",
                self._idx_r,
            )
            return False

        self._configure_cap(self._cap_l, "Left")
        self._configure_cap(self._cap_r, "Right")

        # Warm up: drain a few frames so auto-exposure settles
        log.info("Warming up cameras (10 frames)…")
        for _ in range(10):
            self._cap_l.read()
            self._cap_r.read()

        self._opened = True
        return True

    def read(self) -> FramePair:
        ts = time.monotonic()
        ok_l, frame_l = self._cap_l.read()
        ok_r, frame_r = self._cap_r.read()
        return FramePair(
            left=frame_l if ok_l else None,
            right=frame_r if ok_r else None,
            timestamp=ts,
        )

    def close(self) -> None:
        if self._cap_l:
            self._cap_l.release()
        if self._cap_r:
            self._cap_r.release()
        log.info("Mac cameras released")

    @property
    def is_synced(self) -> bool:
        return False  # software sync only


# ─────────────────────────────────────────────
# Backend 2 — Jetson CSI cameras (GStreamer)
# ─────────────────────────────────────────────

class JetsonCSIStereoCapture(StereoCaptureBase):
    """
    Two CSI cameras on Jetson Orin Nano Super via GStreamer / nvarguscamerasrc.

    Camera modules tested: IMX219 (Raspberry Pi v2 compatible),
    IMX477 (High Quality Camera), or any MIPI CSI-2 module
    supported by Jetson's ISP.

    Sensor IDs
    ----------
    sensor_id=0 → CSI connector J13 (usually left)
    sensor_id=1 → CSI connector J14 (usually right)

    Sync note
    ---------
    nvarguscamerasrc does NOT hardware-sync two sensors by default.
    For production, wire a GPIO trigger to both camera modules'
    XVS (vertical sync) pin and enable sync mode in the ISP.
    Until then, this provides the same software-level sync as Mac.

    Flip
    ----
    flip_method=0  → no flip
    flip_method=2  → 180° (useful if camera is mounted upside-down)
    """

    def __init__(
        self,
        config: CameraConfig,
        sensor_id_left: int = 0,
        sensor_id_right: int = 1,
        flip_method: int = 0,
    ):
        super().__init__(config)
        self._sid_l = sensor_id_left
        self._sid_r = sensor_id_right
        self._flip = flip_method
        self._cap_l: Optional[cv2.VideoCapture] = None
        self._cap_r: Optional[cv2.VideoCapture] = None

    def _gstreamer_pipeline(self, sensor_id: int) -> str:
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} "
            f"! video/x-raw(memory:NVMM), "
            f"width=(int){self.config.width}, "
            f"height=(int){self.config.height}, "
            f"format=(string)NV12, "
            f"framerate=(fraction){self.config.fps}/1 "
            f"! nvvidconv flip-method={self._flip} "
            f"! video/x-raw, width=(int){self.config.width}, "
            f"height=(int){self.config.height}, format=(string)BGRx "
            f"! videoconvert "
            f"! video/x-raw, format=(string)BGR "
            f"! appsink max-buffers=1 drop=True"
        )

    def open(self) -> bool:
        log.info(
            "Opening Jetson CSI cameras (sensor L=%d, R=%d)",
            self._sid_l, self._sid_r,
        )
        pipeline_l = self._gstreamer_pipeline(self._sid_l)
        pipeline_r = self._gstreamer_pipeline(self._sid_r)
        log.debug("Pipeline L: %s", pipeline_l)
        log.debug("Pipeline R: %s", pipeline_r)

        self._cap_l = cv2.VideoCapture(pipeline_l, cv2.CAP_GSTREAMER)
        self._cap_r = cv2.VideoCapture(pipeline_r, cv2.CAP_GSTREAMER)

        if not self._cap_l.isOpened():
            log.error(
                "Could not open CSI camera sensor_id=%d. "
                "Check: sensor is seated, nvarguscamerasrc is available, "
                "run 'nvgstcapture-1.0 --sensor-id=%d' to verify.",
                self._sid_l, self._sid_l,
            )
            return False
        if not self._cap_r.isOpened():
            log.error(
                "Could not open CSI camera sensor_id=%d.",
                self._sid_r,
            )
            return False

        log.info(
            "Jetson CSI cameras opened at %dx%d @ %d fps",
            self.config.width, self.config.height, self.config.fps,
        )
        self._opened = True
        return True

    def read(self) -> FramePair:
        ts = time.monotonic()
        ok_l, frame_l = self._cap_l.read()
        ok_r, frame_r = self._cap_r.read()
        return FramePair(
            left=frame_l if ok_l else None,
            right=frame_r if ok_r else None,
            timestamp=ts,
        )

    def close(self) -> None:
        if self._cap_l:
            self._cap_l.release()
        if self._cap_r:
            self._cap_r.release()
        log.info("Jetson CSI cameras released")

    @property
    def is_synced(self) -> bool:
        # Set True only after GPIO XVS sync is wired and verified
        return False


# ─────────────────────────────────────────────
# Factory — single entry point for the rest of the system
# ─────────────────────────────────────────────

BACKENDS = {
    "mac": MacStereoCapture,
    "jetson": JetsonCSIStereoCapture,
}


def create_capture(
    backend: str,
    config: Optional[CameraConfig] = None,
    **kwargs,
) -> StereoCaptureBase:
    """
    Factory function — instantiate the correct backend by name.

    Parameters
    ----------
    backend : str
        "mac" or "jetson"
    config : CameraConfig, optional
        Resolution and FPS settings. Defaults to 1280×720 @ 30 fps.
    **kwargs
        Passed directly to the backend constructor.
        Mac:    camera_index_left, camera_index_right
        Jetson: sensor_id_left, sensor_id_right, flip_method

    Example
    -------
        cap = create_capture("mac", camera_index_left=0, camera_index_right=1)
        with cap:
            pair = cap.read()
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(BACKENDS)}"
        )
    cfg = config or CameraConfig()
    return BACKENDS[backend](cfg, **kwargs)


# ─────────────────────────────────────────────
# Capture modes
# ─────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, label: str, ts: float) -> np.ndarray:
    """Burn a small label and timestamp onto a frame (non-destructive copy)."""
    out = frame.copy()
    cv2.putText(out, label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(out, f"{ts:.3f}s", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def run_preview(capture: StereoCaptureBase) -> None:
    """
    Live side-by-side preview of both cameras.
    Press Q to quit, S to save a snapshot pair.
    """
    log.info("Preview mode — press Q to quit, S to save snapshot")
    snapshot_dir = Path("./snapshots")
    snapshot_dir.mkdir(exist_ok=True)
    snap_count = 0

    while True:
        pair = capture.read()
        if not pair.both_valid:
            log.warning("Dropped frame at t=%.3f", pair.timestamp)
            continue

        left_disp = _draw_overlay(pair.left, "LEFT", pair.timestamp)
        right_disp = _draw_overlay(pair.right, "RIGHT", pair.timestamp)

        # Stack side by side — resize to same height if cameras differ
        if left_disp.shape[0] != right_disp.shape[0]:
            h = min(left_disp.shape[0], right_disp.shape[0])
            scale_l = h / left_disp.shape[0]
            scale_r = h / right_disp.shape[0]
            left_disp = cv2.resize(left_disp, None, fx=scale_l, fy=scale_l)
            right_disp = cv2.resize(right_disp, None, fx=scale_r, fy=scale_r)

        combined = np.hstack([left_disp, right_disp])
        # Scale down if combined is too wide for screen
        if combined.shape[1] > 1600:
            combined = cv2.resize(combined, None, fx=0.6, fy=0.6)

        cv2.imshow("Stereo Preview (Q=quit  S=snapshot)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            snap_count += 1
            ts_str = f"{pair.timestamp:.3f}".replace(".", "_")
            path_l = snapshot_dir / f"snap_{snap_count:04d}_L_{ts_str}.png"
            path_r = snapshot_dir / f"snap_{snap_count:04d}_R_{ts_str}.png"
            cv2.imwrite(str(path_l), pair.left)
            cv2.imwrite(str(path_r), pair.right)
            log.info("Snapshot %d saved → %s, %s", snap_count, path_l, path_r)

    cv2.destroyAllWindows()


def run_calibration(capture: StereoCaptureBase, output_dir: Path) -> None:
    """
    Calibration capture mode.

    Shows live preview. Press SPACE to capture a chessboard pair,
    Q to finish. Aim for 20–30 varied poses.

    Output
    ------
    output_dir/
      calib_NNNN_L.png
      calib_NNNN_R.png

    After capture, run stereo_calibrate.py (next module) on this directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Calibration mode — aim chessboard at both cameras, "
        "SPACE to capture, Q to quit. Target: 20–30 pairs."
    )
    count = 0

    while True:
        pair = capture.read()
        if not pair.both_valid:
            continue

        left_disp = _draw_overlay(pair.left, f"LEFT  [calib pairs: {count}]", pair.timestamp)
        right_disp = _draw_overlay(pair.right, f"RIGHT [calib pairs: {count}]", pair.timestamp)
        combined = np.hstack([left_disp, right_disp])
        if combined.shape[1] > 1600:
            combined = cv2.resize(combined, None, fx=0.6, fy=0.6)

        cv2.imshow("Calibration capture (SPACE=save  Q=done)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            count += 1
            path_l = output_dir / f"calib_{count:04d}_L.png"
            path_r = output_dir / f"calib_{count:04d}_R.png"
            cv2.imwrite(str(path_l), pair.left)
            cv2.imwrite(str(path_r), pair.right)
            log.info(
                "Pair %d saved — %s | %s", count, path_l.name, path_r.name
            )

    cv2.destroyAllWindows()
    log.info("Calibration capture done — %d pairs saved to %s", count, output_dir)
    if count < 15:
        log.warning(
            "Only %d pairs captured — stereo calibration needs at least 20 "
            "for reliable results. Consider re-running.", count
        )


def run_session(
    capture: StereoCaptureBase,
    output_dir: Path,
    save_every_n: int = 1,
) -> None:
    """
    Session capture mode — records synchronized frame pairs during training.

    Press SPACE to start/stop recording a shooting cycle,
    Q to quit the session.

    Parameters
    ----------
    save_every_n : int
        Save 1 in every N frames to limit disk usage.
        At 30 fps, save_every_n=3 gives 10 fps saved (still plenty for pose).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Session mode — SPACE to start/stop recording a cycle, Q to quit"
    )
    recording = False
    cycle_count = 0
    frame_count = 0
    frame_idx = 0

    while True:
        pair = capture.read()
        if not pair.both_valid:
            continue

        frame_idx += 1
        status = "REC" if recording else "STANDBY"
        left_disp = _draw_overlay(
            pair.left,
            f"LEFT  [{status} cycle={cycle_count} f={frame_count}]",
            pair.timestamp,
        )
        right_disp = _draw_overlay(
            pair.right,
            f"RIGHT [{status} cycle={cycle_count} f={frame_count}]",
            pair.timestamp,
        )
        # Red border while recording
        if recording:
            cv2.rectangle(left_disp, (0, 0), (left_disp.shape[1]-1, left_disp.shape[0]-1),
                          (0, 0, 220), 4)
            cv2.rectangle(right_disp, (0, 0), (right_disp.shape[1]-1, right_disp.shape[0]-1),
                          (0, 0, 220), 4)

        combined = np.hstack([left_disp, right_disp])
        if combined.shape[1] > 1600:
            combined = cv2.resize(combined, None, fx=0.6, fy=0.6)

        cv2.imshow("Session capture (SPACE=start/stop  Q=quit)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            recording = not recording
            if recording:
                cycle_count += 1
                frame_count = 0
                cycle_dir = output_dir / f"cycle_{cycle_count:04d}"
                cycle_dir.mkdir(exist_ok=True)
                log.info("Recording cycle %d → %s", cycle_count, cycle_dir)
            else:
                log.info(
                    "Cycle %d stopped — %d frames saved", cycle_count, frame_count
                )

        if recording and (frame_idx % save_every_n == 0):
            frame_count += 1
            path_l = cycle_dir / f"f{frame_count:05d}_L.png"
            path_r = cycle_dir / f"f{frame_count:05d}_R.png"
            cv2.imwrite(str(path_l), pair.left)
            cv2.imwrite(str(path_r), pair.right)

    cv2.destroyAllWindows()
    log.info(
        "Session done — %d cycles captured to %s", cycle_count, output_dir
    )


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stereo frame capture for archery 3D pose system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend", choices=list(BACKENDS), default="mac",
        help="Camera backend to use",
    )
    p.add_argument(
        "--mode", choices=["preview", "calibration", "session"], default="preview",
        help=(
            "preview: live view only | "
            "calibration: capture chessboard pairs | "
            "session: record shooting cycles"
        ),
    )
    p.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Directory for saved frames (calibration / session modes)",
    )
    # Resolution / FPS
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    # Mac-specific
    p.add_argument(
        "--cam-left", type=int, default=0,
        help="[mac] OpenCV camera index for left camera",
    )
    p.add_argument(
        "--cam-right", type=int, default=1,
        help="[mac] OpenCV camera index for right camera",
    )
    # Jetson-specific
    p.add_argument(
        "--sensor-left", type=int, default=0,
        help="[jetson] CSI sensor ID for left camera",
    )
    p.add_argument(
        "--sensor-right", type=int, default=1,
        help="[jetson] CSI sensor ID for right camera",
    )
    p.add_argument(
        "--flip", type=int, default=0,
        help="[jetson] nvvidconv flip-method (0=none, 2=180°)",
    )
    # Session options
    p.add_argument(
        "--save-every", type=int, default=1,
        help="[session] Save 1 in every N frames (e.g. 3 = 10 fps saved at 30 fps input)",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = CameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    # Build backend-specific kwargs
    if args.backend == "mac":
        kwargs = {
            "camera_index_left": args.cam_left,
            "camera_index_right": args.cam_right,
        }
    else:  # jetson
        kwargs = {
            "sensor_id_left": args.sensor_left,
            "sensor_id_right": args.sensor_right,
            "flip_method": args.flip,
        }

    log.info(
        "Backend: %s | Mode: %s | Resolution: %dx%d @ %d fps",
        args.backend, args.mode, config.width, config.height, config.fps,
    )

    try:
        capture = create_capture(args.backend, config, **kwargs)
        with capture:
            if not capture.is_synced:
                log.warning(
                    "Software sync only — small temporal offset between L and R frames. "
                    "Acceptable for development; use hardware GPIO sync on Jetson for production."
                )
            if args.mode == "preview":
                run_preview(capture)
            elif args.mode == "calibration":
                run_calibration(capture, args.output)
            elif args.mode == "session":
                run_session(capture, args.output, save_every_n=args.save_every)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user")


if __name__ == "__main__":
    main()