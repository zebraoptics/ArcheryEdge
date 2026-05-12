import cv2
from utils import DualCameraConfig, CameraConfig, DualCSICamera


# Apply 2x2 binning for higher frame rates at the cost of resolution. (Only for the IMX219 sensor.)
config = DualCameraConfig(
    cam0=CameraConfig(sensor_id=0, width=1640, height=1232, framerate=30,
                      sensor_mode=3, rotation=270),
    cam1=CameraConfig(sensor_id=1, width=1640, height=1232, framerate=30,
                      sensor_mode=3, rotation=90),
)

with DualCSICamera(config) as cams:
    print(f"Streaming — press 'q' to quit, 's' to save a snapshot pair.")
    while True:
        left, right = cams.read_sync(timeout=2.0, max_dt=0.020)

        if left is None or right is None:
            continue

        # Frames arrive upright thanks to CameraConfig.rotation.
        h, w = left.shape[:2]
        disp_left  = cv2.resize(left,  (480, 480 * h // w))
        disp_right = cv2.resize(right, (480, 480 * h // w))

        # Overlay FPS
        for img, cam in [(disp_left, cams.cam0), (disp_right, cams.cam1)]:
            cv2.putText(img, f"cam{cam.sensor_id}  {cam.fps:.1f} fps",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = cv2.hconcat([disp_left, disp_right])
        cv2.imshow("JetsonEyes — Dual CSI", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            tag = f"{int(__import__('time').time())}"
            cv2.imwrite(f"snap_cam0_{tag}.jpg", left)
            cv2.imwrite(f"snap_cam1_{tag}.jpg", right)
            print(f"Saved snapshot pair: snap_cam*_{tag}.jpg")

cv2.destroyAllWindows()