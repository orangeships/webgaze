import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

from model import Model
import mediapipe as mp

MODEL_PATH = 'pretrain/GazeTR-H-ETH.pt'
DEVICE = 'cuda'
CAMERA_INDEX = 1
INPUT_SIZE = 224
ARROW_SCALE = 120
MEDIAN_FILTER_SIZE = 5
FACE_BOX_MARGIN = 0.15
FACE_BOX_UP_SHIFT = 0.08
YAW_BIAS = -0.28659505
PITCH_BIAS = -0.824546585

_FACE_MESH = mp.solutions.face_mesh
_TRANSFORM = transforms.ToTensor()


def gaze_to_3d(yaw, pitch):
    vec = np.zeros(3, dtype=np.float32)
    vec[0] = -np.cos(pitch) * np.sin(yaw)
    vec[1] = -np.sin(pitch)
    vec[2] = -np.cos(pitch) * np.cos(yaw)
    return vec


def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec
    return vec / norm


def _center_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    x_min = (w - side) // 2
    y_min = (h - side) // 2
    y_min = int(max(0, y_min - side * FACE_BOX_UP_SHIFT))
    cropped = frame[y_min:y_min + side, x_min:x_min + side]
    cropped = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))
    return cropped, (x_min, y_min, side, side)


def detect_and_crop_face(frame, face_mesh):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return _center_crop(frame)

    face_landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    xs = np.array([lm.x for lm in face_landmarks.landmark], dtype=np.float32)
    ys = np.array([lm.y for lm in face_landmarks.landmark], dtype=np.float32)

    x_min = np.clip(xs.min() * w, 0, w - 1)
    x_max = np.clip(xs.max() * w, 0, w - 1)
    y_min = np.clip(ys.min() * h, 0, h - 1)
    y_max = np.clip(ys.max() * h, 0, h - 1)

    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 2 or box_h < 2:
        return _center_crop(frame)

    side = max(box_w, box_h) * (1.0 + 2.0 * FACE_BOX_MARGIN)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cy = cy - side * FACE_BOX_UP_SHIFT

    x_min = int(max(0, cx - side / 2.0))
    x_max = int(min(w, cx + side / 2.0))
    y_min = int(max(0, cy - side / 2.0))
    y_max = int(min(h, cy + side / 2.0))

    if x_max <= x_min or y_max <= y_min:
        return _center_crop(frame)

    cropped = frame[y_min:y_max, x_min:x_max]
    cropped = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    return cropped, bbox


def preprocess_frame(frame, face_mesh):
    cropped, bbox = detect_and_crop_face(frame, face_mesh)
    tensor = _TRANSFORM(cropped).unsqueeze(0)
    return tensor, bbox


def draw_gaze_arrow(image, bbox, yaw, pitch):
    if bbox is None:
        x_min = 0
        y_min = 0
        x_max = image.shape[1]
        y_max = image.shape[0]
    else:
        x, y, width, height = bbox
        x_min = x
        y_min = y
        x_max = x + width
        y_max = y + height

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    gaze_vec = normalize_vec(gaze_to_3d(yaw, pitch))
    dx = int(ARROW_SCALE * gaze_vec[0])
    dy = int(ARROW_SCALE * gaze_vec[1])

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)

    cv2.circle(image, (x_center, y_center), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.arrowedLine(
        image,
        point1,
        point2,
        color=(0, 255, 0),
        thickness=3,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )

    if bbox is not None:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return image


def run_webcam(model, device):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise ValueError('Failed to open camera')

    gaze_buffer = deque(maxlen=MEDIAN_FILTER_SIZE)
    face_mesh = _FACE_MESH.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            img_tensor, bbox = preprocess_frame(frame, face_mesh)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                gaze = model({'face': img_tensor})

            gaze = gaze.cpu().numpy()[0]
            gaze_buffer.append(gaze)
            gaze_smoothed = np.median(np.stack(gaze_buffer, axis=0), axis=0)

            yaw = float(gaze_smoothed[0]) - YAW_BIAS
            pitch = float(gaze_smoothed[1]) - PITCH_BIAS

            result_img = draw_gaze_arrow(frame.copy(), bbox, yaw, pitch)

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            cv2.putText(
                result_img,
                f"{frame_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('GazeTR Webcam', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    device = DEVICE
    if device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA not available, using CPU.')
        device = 'cpu'

    model = Model()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    run_webcam(model, device)


if __name__ == '__main__':
    main()
