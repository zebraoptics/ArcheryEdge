"""
Keypoint smoothing filters for pose estimation pipelines.

All smoothers share a common interface::

    smoothed_xy = smoother.smooth(xy)

where ``xy`` is ``np.ndarray`` of shape ``(K, 2)`` — K keypoints, each as
``(x, y)`` in pixel or normalised coordinates — and the return value has the
same shape.

Filter state is initialised lazily on the first call so ``n_landmarks`` does
not need to be specified at construction time.

Usage
-----
::

    from reconstruction.filters import make_smoother

    smoother = make_smoother('one_euro', fps=30.0)
    smoothed_xy = smoother.smooth(raw_xy)   # raw_xy: np.ndarray (K, 2)
    good luck!
"""

import math
import numpy as np

# COCO arm keypoint indices: left/right shoulder, elbow, wrist
COCO_ARM_INDICES = [5, 6, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# Base smoothers
# ---------------------------------------------------------------------------

class NoSmoother:
    """Pass-through — returns the input unchanged."""

    def smooth(self, xy: np.ndarray) -> np.ndarray:
        return xy


class EMASmoother:
    """
    Exponential Moving Average smoother.

    Parameters
    ----------
    alpha : float
        Blending weight for the new observation (0 < alpha ≤ 1).
        Higher = more responsive, lower = smoother.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._prev = None

    def smooth(self, xy: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = xy.copy()
            return xy.copy()
        smoothed = self.alpha * xy + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed


class OneEuroSmoother:
    """
    1€ filter — adapts its cutoff frequency to signal speed.

    Slow motion  → low cutoff  → strong smoothing.
    Fast motion  → high cutoff → low lag.

    Parameters
    ----------
    fps       : float  — frames per second of the video
    mincutoff : float  — minimum cutoff frequency in Hz
    beta      : float  — speed coefficient (higher = less lag during fast motion)
    dcutoff   : float  — cutoff for the derivative low-pass filter
    """

    def __init__(self, fps: float, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.fps = fps
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def smooth(self, xy: np.ndarray) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev = xy.copy()
            self._dx_prev = np.zeros_like(xy)
            return xy.copy()

        dt = 1.0 / self.fps
        dx = (xy - self._x_prev) * self.fps
        a_d = self._alpha(dt, self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        r = 2 * math.pi * cutoff * dt
        a = r / (r + 1)

        smoothed = a * xy + (1 - a) * self._x_prev
        self._x_prev = smoothed
        self._dx_prev = dx_hat
        return smoothed

    def reset_to(self, xy: np.ndarray) -> None:
        """Snap filter state to ``xy`` — use on abrupt mode transitions."""
        self._x_prev = xy.copy()
        self._dx_prev = np.zeros_like(xy)


class KalmanSmoother:
    """
    Constant-velocity Kalman filter for pose keypoints.

    State per coordinate: ``[position, velocity]``.
    Observation: position only.
    All K×2 coordinates are filtered independently (vectorised).

    Parameters
    ----------
    process_noise     : float — Q scaling; higher = more reactive to detections
    measurement_noise : float — R scaling; higher = smoother but more lag
    """

    def __init__(self, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # State arrays are allocated lazily on first call
        self._n = None
        self._x_hat = None
        self._P = None
        self._F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._H = np.array([1.0, 0.0])
        self._Q = process_noise * np.array([[0.25, 0.5], [0.5, 1.0]])
        self._R = measurement_noise

    def _init(self, n: int) -> None:
        self._n = n
        self._x_hat = np.zeros((n, 2))
        self._P = np.tile(np.eye(2), (n, 1, 1))

    def smooth(self, xy: np.ndarray) -> np.ndarray:
        flat = xy.flatten()          # (K*2,)
        n = len(flat)

        if self._n is None:
            self._init(n)
            self._x_hat[:, 0] = flat
            return xy.copy()

        # Predict
        x_p = self._x_hat @ self._F.T                   # (n, 2)
        P_p = self._F @ self._P @ self._F.T + self._Q   # (n, 2, 2)

        # Update
        innov = flat - x_p[:, 0]                        # (n,)
        S = P_p[:, 0, 0] + self._R                      # (n,)
        K = P_p[:, :, 0] / S[:, np.newaxis]             # (n, 2)
        self._x_hat = x_p + K * innov[:, np.newaxis]
        KH = np.einsum('ni,j->nij', K, self._H)         # (n, 2, 2)
        self._P = (np.eye(2)[np.newaxis] - KH) @ P_p

        return self._x_hat[:, 0].reshape(xy.shape)


# ---------------------------------------------------------------------------
# Dynamic (speed-adaptive) smoother
# ---------------------------------------------------------------------------

class DynamicSmoother:
    """
    Speed-adaptive filter: applies 1€ filter when arm joints move slowly,
    bypasses filtering when they move fast.

    The 1€ filter is always advanced so its state stays current; on a
    fast→slow transition the filter is snapped to the raw position to
    prevent lag carry-over.

    Parameters
    ----------
    fps              : float  — frames per second
    speed_threshold  : float  — pixels/frame above which no filter is applied
    mincutoff        : float  — 1€ minimum cutoff frequency (Hz)
    beta             : float  — 1€ speed coefficient
    arm_indices      : list   — keypoint indices used for speed measurement
                               (default: COCO shoulders/elbows/wrists)
    """

    def __init__(self, fps: float, speed_threshold: float = 20.0,
                 mincutoff: float = 1.0, beta: float = 0.007,
                 arm_indices: list = None):
        self.fps = fps
        self.speed_threshold = speed_threshold
        self.arm_indices = arm_indices if arm_indices is not None else COCO_ARM_INDICES
        self._filter = OneEuroSmoother(fps, mincutoff=mincutoff, beta=beta)
        self._prev_raw = None
        self.mode = 'one_euro'   # current active mode; readable for logging

    def arm_speed(self, xy: np.ndarray) -> float:
        """Mean Euclidean displacement of arm joints since last frame (px/frame)."""
        if self._prev_raw is None:
            return 0.0
        disp = xy[self.arm_indices] - self._prev_raw[self.arm_indices]
        return float(np.linalg.norm(disp, axis=1).mean())

    def smooth(self, xy: np.ndarray) -> np.ndarray:
        speed = self.arm_speed(xy)
        self._prev_raw = xy.copy()

        # Always advance 1€ filter to keep state fresh
        filtered = self._filter.smooth(xy)

        if speed > self.speed_threshold:
            self.mode = 'none'
            self._filter.reset_to(xy)
            return xy.copy()
        else:
            self.mode = 'one_euro'
            return filtered


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_smoother(filter_type: str, fps: float = 30.0, **kwargs):
    """
    Create a smoother by name.

    Parameters
    ----------
    filter_type : 'none' | 'ema' | 'one_euro' | 'kalman' | 'dynamic'
    fps         : frames per second (required by one_euro, kalman, dynamic)
    **kwargs    : forwarded to the smoother constructor

    Returns
    -------
    Smoother instance with a ``smooth(xy: np.ndarray) -> np.ndarray`` method.
    """
    if filter_type == 'none':
        return NoSmoother()
    elif filter_type == 'ema':
        return EMASmoother(**kwargs)
    elif filter_type == 'one_euro':
        return OneEuroSmoother(fps=fps, **kwargs)
    elif filter_type == 'kalman':
        return KalmanSmoother(**kwargs)
    elif filter_type == 'dynamic':
        return DynamicSmoother(fps=fps, **kwargs)
    else:
        raise ValueError(f"Unknown filter type '{filter_type}'. "
                         f"Choose from: none, ema, one_euro, kalman, dynamic")
