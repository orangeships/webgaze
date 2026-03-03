#!/usr/bin/env python3
"""
Visual comparison: model.py vs modeltr.py (left/right split),
with per-frame timing and runtime averages.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from gaze_tracking.model import EyeModel as ModelBaseline
from gaze_tracking.modeltr import EyeModel as ModelGazeTR


def _extract_eye_info(result):
    if result is None:
        return None
    if isinstance(result, (tuple, list)):
        return result[0] if len(result) > 0 else None
    return result


def main():
    print("Gaze model compare: model.py vs modeltr.py")
    print("=" * 60)

    try:
        print("Initializing models...")
        base_model = ModelBaseline(".")
        tr_model = ModelGazeTR(".")
        print("Models ready.")

        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            return

        print(f"Camera {camera_index} opened.")
        window_name = "Gaze Model Compare (Left: model.py | Right: modeltr.py)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        frame_count = 0
        base_times = []
        tr_times = []

        print("Press 'q' to quit, 's' to save a frame.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            h, w = frame.shape[:2]
            comparison_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)

            frame_base = frame.copy()
            start_time1 = time.time()
            result_base = base_model.get_gaze(frame_base, imshow=False)
            process_time1 = time.time() - start_time1
            base_times.append(process_time1)
            gaze_result_base = _extract_eye_info(result_base)

            frame_tr = frame.copy()
            start_time2 = time.time()
            result_tr = tr_model.get_gaze(frame_tr, imshow=False)
            process_time2 = time.time() - start_time2
            tr_times.append(process_time2)
            gaze_result_tr = _extract_eye_info(result_tr)

            comparison_frame[:, :w] = frame_base
            cv2.putText(
                comparison_frame,
                "model.py",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if gaze_result_base is not None:
                gaze_vec1 = gaze_result_base["gaze"]
                cv2.putText(
                    comparison_frame,
                    f"Gaze: [{gaze_vec1[0]:.3f}, {gaze_vec1[1]:.3f}]",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    comparison_frame,
                    f"Eye State: {gaze_result_base['EyeState']}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    comparison_frame,
                    "No face detected",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            avg_time1 = np.mean(base_times[-10:]) if len(base_times) >= 10 else np.mean(base_times)
            cv2.putText(
                comparison_frame,
                f"Time: {process_time1*1000:.1f}ms (avg {avg_time1*1000:.1f}ms)",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            comparison_frame[:, w:] = frame_tr
            cv2.putText(
                comparison_frame,
                "modeltr.py",
                (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            if gaze_result_tr is not None:
                gaze_vec2 = gaze_result_tr["gaze"]
                cv2.putText(
                    comparison_frame,
                    f"Gaze: [{gaze_vec2[0]:.3f}, {gaze_vec2[1]:.3f}]",
                    (w + 10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    comparison_frame,
                    f"Eye State: {gaze_result_tr['EyeState']}",
                    (w + 10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
            else:
                cv2.putText(
                    comparison_frame,
                    "No face detected",
                    (w + 10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            avg_time2 = np.mean(tr_times[-10:]) if len(tr_times) >= 10 else np.mean(tr_times)
            cv2.putText(
                comparison_frame,
                f"Time: {process_time2*1000:.1f}ms (avg {avg_time2*1000:.1f}ms)",
                (w + 10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            speed_ratio = process_time2 / process_time1 if process_time1 > 0 else 1.0
            cv2.putText(
                comparison_frame,
                f"Speed ratio: {speed_ratio:.2f}x",
                (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                comparison_frame,
                f"Frame: {frame_count}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, comparison_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                out_name = f"comparison_frame_{frame_count}.jpg"
                cv2.imwrite(out_name, comparison_frame)
                print(f"Saved frame: {out_name}")

        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

    except Exception as exc:
        print(f"Test error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
