"""
JetsonEyes — Dual CSI Camera Library
Streams from two MIPI CSI cameras on Jetson Orin Nano Super using
GStreamer nvarguscamerasrc → OpenCV VideoCapture.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single CSI camera."""
    sensor_id: int = 0
    width: int = 1920
    height: int = 1080
    framerate: int = 30
    flip_method: int = 0   # 0=none, 1=ccw90, 2=rot180, 3=cw90, 4=horiz, 5=ul-lr, 6=vert, 7=ur-ll
    sensor_mode: int = -1  # IMX219: -1=auto, 0=3280x2464, 1=3280x1848, 2=1920x1080,
                           #         3=1640x1232(2x2bin), 4=1640x922(2x2bin),
                           #         5=1280x720(2x2bin@60), 6=1280x720(2x2bin@120)
    # Clockwise rotation in degrees applied to every frame returned by read().
    # Use this to make the camera output upright when the sensor is physically
    # mounted on its side. Distinct from flip_method (which is a GStreamer
    # pipeline option for in-driver transforms). All downstream consumers —
    # calibration, stereo, application — see post-rotation frames.
    rotation: int = 0      # 0, 90, 180, or 270


def _gstreamer_pipeline(cfg: CameraConfig) -> str:
    mode_str = f"sensor-mode={cfg.sensor_mode} " if cfg.sensor_mode >= 0 else ""
    return (
        f"nvarguscamerasrc sensor-id={cfg.sensor_id} {mode_str}! "
        f"video/x-raw(memory:NVMM), width={cfg.width}, height={cfg.height}, "
        f"format=NV12, framerate={cfg.framerate}/1 ! "
        f"nvvidconv flip-method={cfg.flip_method} ! "
        f"video/x-raw, width={cfg.width}, height={cfg.height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )


class CSICamera:
    """Single CSI camera stream."""

    _ROTATION_CODES = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(self, config: CameraConfig):
        if config.rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"CameraConfig.rotation must be 0, 90, 180, or 270 — got {config.rotation}"
            )
        self.config = config
        self._rotate_code = self._ROTATION_CODES.get(config.rotation)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_time: Optional[float] = None  # monotonic timestamp of latest frame
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._fps = 0.0

    @property
    def sensor_id(self) -> int:
        return self.config.sensor_id

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        """Open the camera and start the capture thread."""
        pipeline = _gstreamer_pipeline(self.config)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Camera {self.config.sensor_id}: failed to open GStreamer pipeline.\n"
                f"Pipeline: {pipeline}"
            )
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._rotate_code is not None:
                frame = cv2.rotate(frame, self._rotate_code)
            now = time.monotonic()
            with self._lock:
                self._frame = frame
                self._frame_time = now
                self._frame_count += 1
            # FPS calculation
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame (BGR), or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def read_with_timestamp(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return (frame, monotonic_timestamp), both None if not yet available."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), self._frame_time

    def close(self) -> None:
        """Stop the capture thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class DualCameraConfig:
    """Configuration for the dual-camera setup."""
    cam0: CameraConfig = field(default_factory=lambda: CameraConfig(sensor_id=0))
    cam1: CameraConfig = field(default_factory=lambda: CameraConfig(sensor_id=1))


class DualCSICamera:
    """
    Manages two CSI cameras simultaneously.

    Usage
    -----
    with DualCSICamera() as cams:
        while True:
            left, right = cams.read()
            if left is not None and right is not None:
                cv2.imshow("left", left)
                cv2.imshow("right", right)
            if cv2.waitKey(1) == ord('q'):
                break
    """

    def __init__(self, config: Optional[DualCameraConfig] = None):
        cfg = config or DualCameraConfig()
        self._cam0 = CSICamera(cfg.cam0)
        self._cam1 = CSICamera(cfg.cam1)

    @property
    def cam0(self) -> CSICamera:
        return self._cam0

    @property
    def cam1(self) -> CSICamera:
        return self._cam1

    def open(self) -> None:
        """Open both cameras. Raises RuntimeError if either fails."""
        self._cam0.open()
        self._cam1.open()

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (cam0_frame, cam1_frame). Either may be None if not yet ready."""
        return self._cam0.read(), self._cam1.read()

    def read_sync(
        self,
        timeout: float = 1.0,
        max_dt: float = 0.010,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a temporally-matched frame pair using software synchronisation.

        The method waits until both cameras have a *fresh* frame whose
        capture timestamps differ by no more than ``max_dt`` seconds.
        When only one camera has advanced, it waits for the other to catch
        up rather than returning a stale pair.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait before raising TimeoutError.
        max_dt:
            Maximum allowed timestamp difference between the two frames (s).
            Default 10 ms is tight enough for 30 fps (frame period ≈ 33 ms)
            while tolerating normal OS scheduling jitter.

        Returns
        -------
        (cam0_frame, cam1_frame) as BGR numpy arrays.

        Raises
        ------
        TimeoutError
            If a synchronised pair cannot be obtained within ``timeout`` s.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            f0, t0 = self._cam0.read_with_timestamp()
            f1, t1 = self._cam1.read_with_timestamp()

            if f0 is None or f1 is None:
                time.sleep(0.001)
                continue

            dt = abs(t0 - t1)
            if dt <= max_dt:
                return f0, f1

            # Stale camera: wait for the lagging one to produce a new frame
            # before sampling again — avoids busy-spinning on the same pair.
            time.sleep(min(dt * 0.5, 0.005))

        raise TimeoutError(
            f"Timed out waiting for synchronised frame pair within {max_dt*1000:.1f} ms."
        )

    def close(self) -> None:
        self._cam0.close()
        self._cam1.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
