import argparse
import os
import sys
import time

import cv2

from gaze_tracking.model import EyeModel


def _extract_eye_info(result):
    if result is None:
        return None
    if isinstance(result, (tuple, list)):
        return result[0] if len(result) > 0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description="Gaze preview window")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--window-w", type=int, default=360)
    parser.add_argument("--window-h", type=int, default=270)
    parser.add_argument("--fps", type=float, default=15)
    args = parser.parse_args()

    if args.project_dir:
        project_dir = args.project_dir
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = EyeModel(project_dir)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[GazePreview] Failed to open camera {args.camera_index}")
        return 1

    window_name = "Gaze Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.window_w > 0 and args.window_h > 0:
        cv2.resizeWindow(window_name, args.window_w, args.window_h)

    frame_delay = 1.0 / max(1.0, args.fps)
    last_tick = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = model.get_gaze(frame=frame, imshow=False)
        except Exception as exc:
            cv2.putText(
                frame,
                f"Gaze error: {exc}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            eye_info = None
        else:
            eye_info = _extract_eye_info(result)
            if eye_info is None:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            else:
                gaze_vec = eye_info.get("gaze")
                if gaze_vec is not None:
                    cv2.putText(
                        frame,
                        f"Gaze: [{gaze_vec[0]:.3f}, {gaze_vec[1]:.3f}]",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                eye_state = eye_info.get("EyeState")
                if eye_state is not None:
                    cv2.putText(
                        frame,
                        f"Eye State: {eye_state}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        display_frame = frame
        if args.window_w > 0 and args.window_h > 0:
            display_frame = cv2.resize(
                frame, (args.window_w, args.window_h), interpolation=cv2.INTER_AREA
            )
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        now = time.time()
        sleep_time = frame_delay - (now - last_tick)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_tick = time.time()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
