import os
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp

import utilities.utils as util
from gazetr.resnet import resnet18


MODEL_FILENAME = "GazeTR-H-ETH.pt"
INPUT_SIZE = 224
FACE_BOX_MARGIN = 0.15
FACE_BOX_UP_SHIFT = 0.08
YAW_BIAS = -0.28659505
PITCH_BIAS = -0.824546585
MEDIAN_FILTER_SIZE = 5


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


def to_relative_coordinates(absolute_boxes, absolute_centers, face_box):
    face_xmin, face_ymin, _, _ = face_box

    relative_eye_boxes = []
    for eye_box in absolute_boxes:
        xmin, ymin, xmax, ymax = eye_box
        relative_eye_boxes.append([
            xmin - face_xmin, ymin - face_ymin,
            xmax - face_xmin, ymax - face_ymin,
        ])

    relative_eye_centers = []
    for eye_center in absolute_centers:
        cx, cy = eye_center
        relative_eye_centers.append([
            cx - face_xmin, cy - face_ymin,
        ])

    return relative_eye_boxes, relative_eye_centers


def _square_face_box(face_box, frame_shape):
    x_min, y_min, x_max, y_max = face_box
    h, w = frame_shape[:2]

    box_w = max(2, x_max - x_min)
    box_h = max(2, y_max - y_min)
    side = max(box_w, box_h) * (1.0 + 2.0 * FACE_BOX_MARGIN)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cy -= side * FACE_BOX_UP_SHIFT

    x_min = int(max(0, cx - side / 2.0))
    x_max = int(min(w, cx + side / 2.0))
    y_min = int(max(0, cy - side / 2.0))
    y_max = int(min(h, cy + side / 2.0))

    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class GazeTRNet(nn.Module):
    def __init__(self):
        super().__init__()
        maps = 32
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6

        self.base_model = resnet18(pretrained=False, maps=maps)

        encoder_layer = TransformerEncoderLayer(
            maps,
            nhead,
            dim_feedforward,
            dropout,
        )
        encoder_norm = nn.LayerNorm(maps)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.pos_embedding = nn.Embedding(dim_feature + 1, maps)
        self.feed = nn.Linear(maps, 2)
        self.loss_op = nn.L1Loss()

    def forward(self, x_in):
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)

        cls = self.cls_token.repeat((1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)

        position = torch.arange(
            0,
            self.pos_embedding.num_embeddings,
            device=feature.device,
        )
        pos_feature = self.pos_embedding(position)

        feature = self.encoder(feature, pos_feature)
        feature = feature.permute(1, 2, 0)
        feature = feature[:, :, 0]
        gaze = self.feed(feature)
        return gaze


class MediaPipeFace:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.75,
        )

        self.l_eye_8 = [33, 133, 160, 159, 158, 144, 145, 153]
        self.r_eye_8 = [362, 263, 385, 386, 387, 373, 374, 380]

        self.stable_indices = [
            10, 151, 9, 168, 6, 1, 152, 103, 332, 54, 284, 21, 251, 127,
            356, 133, 362, 33, 263, 234, 454, 93, 323, 58, 288, 197, 5, 2,
            164, 172, 397, 176, 400, 148, 377,
        ]

        self._init_pnp_parameters()

    def _init_pnp_parameters(self):
        self.pnp_indices = [
            10, 151, 9, 168, 6, 1, 152, 103, 332, 54, 284, 21, 251, 127,
            356, 133, 362, 33, 263, 234, 454, 93, 323, 58, 288, 197, 5, 2,
            164, 172, 397, 176, 400, 148, 377,
        ]

        self.model_points = np.array([
            [0.0, 112.7, 36.8],
            [0.0, 77.5, 59.8],
            [0.0, 55.2, 63.8],
            [0.0, 41.5, 65.4],
            [0.0, 21.0, 70.8],
            [0.0, -12.4, 93.9],
            [0.0, -103.5, 23.9],
            [-44.3, 75.3, 40.8],
            [44.3, 75.3, 40.8],
            [-23.6, 68.3, 56.4],
            [23.6, 68.3, 56.4],
            [-45.5, 45.4, 21.9],
            [45.5, 45.4, 21.9],
            [-75.4, 18.2, -14.6],
            [75.4, 18.2, -14.6],
            [-15.1, 15.5, 44.4],
            [15.1, 15.5, 44.4],
            [-46.1, 14.8, 30.6],
            [46.1, 14.8, 30.6],
            [-82.9, -15.1, -12.1],
            [82.9, -15.1, -12.1],
            [-43.9, -35.2, 28.5],
            [43.9, -35.2, 28.5],
            [-47.9, -46.7, 32.3],
            [47.9, -46.7, 32.3],
            [0.0, 11.2, 75.8],
            [0.0, -1.9, 81.3],
            [0.0, -32.8, 77.0],
            [0.0, -51.3, 62.4],
            [-18.7, -76.8, 48.0],
            [18.7, -76.8, 48.0],
            [-13.7, -88.1, 40.0],
            [13.7, -88.1, 40.0],
            [-34.2, -87.8, 20.3],
            [34.2, -87.8, 20.3],
        ], dtype=np.float64)

        self.cam_matrix = np.array([
            [1408.572643, 0.000000, 918.986646],
            [0.000000, 1378.049178, 553.273009],
            [0.000000, 0.000000, 1.000000],
        ], dtype=np.float64)
        self.dist_coeffs = np.array([
            [-0.355882, 0.159152, 0.002311, -0.001401, -0.055078],
        ], dtype=np.float64)

        self.nominal_width = 1920
        self.nominal_height = 1080

        self.actual_width = None
        self.actual_height = None

        self.scaled_cam_matrix = None
        self.scaled_dist_coeffs = None

        self.rvec = None
        self.tvec = None

    def update_camera_params(self, frame_width, frame_height):
        if frame_width == self.nominal_width and frame_height == self.nominal_height:
            self.scaled_cam_matrix = self.cam_matrix.copy()
            self.scaled_dist_coeffs = self.dist_coeffs.copy()
        else:
            scale_x = frame_width / self.nominal_width
            scale_y = frame_height / self.nominal_height

            self.scaled_cam_matrix = self.cam_matrix.copy()
            self.scaled_cam_matrix[0, 0] *= scale_x
            self.scaled_cam_matrix[1, 1] *= scale_y
            self.scaled_cam_matrix[0, 2] *= scale_x
            self.scaled_cam_matrix[1, 2] *= scale_y

            self.scaled_dist_coeffs = self.dist_coeffs.copy()

        self.actual_width = frame_width
        self.actual_height = frame_height

    def get_pose(self, frame):
        h, w, _ = frame.shape
        if self.actual_width != w or self.actual_height != h:
            self.update_camera_params(w, h)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            face_landmarks = np.array([
                [landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                for idx in self.pnp_indices
            ], dtype=np.float32)

            success, rvec, tvec, _ = cv2.solvePnPRansac(
                self.model_points,
                face_landmarks,
                self.scaled_cam_matrix,
                self.scaled_dist_coeffs,
                rvec=self.rvec,
                tvec=self.tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if success:
                for _ in range(10):
                    success, rvec, tvec = cv2.solvePnP(
                        self.model_points,
                        face_landmarks,
                        self.scaled_cam_matrix,
                        self.scaled_dist_coeffs,
                        rvec=rvec,
                        tvec=tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                self.rvec, self.tvec = rvec, tvec
                rmat = cv2.Rodrigues(rvec)[0]
                return tvec, rmat

        return None, None

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        pts = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in face_landmarks.landmark
        ])

        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        face_box = [int(xmin), int(ymin), int(xmax), int(ymax)]

        left_eye_center = pts[self.l_eye_8].mean(axis=0).astype(int)
        right_eye_center = pts[self.r_eye_8].mean(axis=0).astype(int)

        def get_eye_box(eye_center, eye_points):
            eye_pts = pts[eye_points]
            x_min, y_min = eye_pts.min(axis=0)
            x_max, y_max = eye_pts.max(axis=0)

            eye_width = max(x_max - x_min, 10)
            eye_height = max(y_max - y_min, 6)
            center_x, center_y = eye_center

            new_width = max(int(eye_width * 1.5), 20)
            new_height = max(int(eye_height * 1.5), 20)

            xmin_eye = max(0, center_x - new_width // 2)
            ymin_eye = max(0, center_y - new_height // 2)
            xmax_eye = min(w, center_x + new_width // 2)
            ymax_eye = min(h, center_y + new_height // 2)

            return [int(xmin_eye), int(ymin_eye), int(xmax_eye), int(ymax_eye)]

        right_eye_box = get_eye_box(right_eye_center, self.r_eye_8)
        left_eye_box = get_eye_box(left_eye_center, self.l_eye_8)
        eye_boxes = [right_eye_box, left_eye_box]
        eye_centers = [right_eye_center.tolist(), left_eye_center.tolist()]

        if len(self.stable_indices) >= 35:
            selected_indices = self.stable_indices[:35]
        else:
            selected_indices = self.stable_indices + [
                i for i in range(468) if i not in self.stable_indices
            ][:35 - len(self.stable_indices)]

        landmarks_35 = pts[selected_indices]
        return face_box, eye_boxes, eye_centers, landmarks_35


class EyeModel:
    def __init__(self, directory, model_path=None, device="AUTO"):
        self.dir = directory
        self.device = self._select_device(device)
        self.model_path = self._resolve_model_path(model_path, directory)
        self.model = GazeTRNet()
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.ToTensor()
        self.mp_face = MediaPipeFace()
        self.queue_gaze = np.nan * np.zeros((3, MEDIAN_FILTER_SIZE))

    def _select_device(self, device):
        if device is None:
            device = "AUTO"
        device_name = str(device).upper()
        if device_name in ("AUTO", "CUDA", "GPU"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if device_name == "CPU":
            return torch.device("cpu")
        return torch.device(device)

    def _resolve_model_path(self, model_path, directory):
        if model_path and os.path.isfile(model_path):
            return model_path

        candidates = [
            os.path.join(directory, "src", "gazetr", MODEL_FILENAME),
            os.path.join(directory, "gazetr", MODEL_FILENAME),
            os.path.join(directory, "pretrain", MODEL_FILENAME),
            os.path.join(os.path.dirname(__file__), "..", "gazetr", MODEL_FILENAME),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.normpath(path)

        raise FileNotFoundError(
            f"Could not find {MODEL_FILENAME}. Set model_path explicitly."
        )

    def get_crop_image(self, image, box):
        xmin, ymin, xmax, ymax = box

        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))

        if xmax <= xmin or ymax <= ymin:
            return None

        crop_image = image[ymin:ymax, xmin:xmax]
        if crop_image is None or crop_image.size == 0:
            return None
        return crop_image

    def draw_eye_line(self, image, face_box, eye_boxes, eye_centers, gaze_x, gaze_y):
        xmin, ymin, xmax, ymax = face_box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

        for eye_box in eye_boxes:
            xmin2, ymin2, xmax2, ymax2 = eye_box
            cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), (255, 255, 255), 1)

        for x, y in eye_centers:
            start = (x, y)
            end = (x + int(gaze_x * 90), y - int(gaze_y * 90))
            cv2.arrowedLine(image, start, end, (0, 0, 255), 2)

    def get_gaze(self, frame, imshow=False, face_boxes=None):
        eye_info = None
        landmarks_35 = None

        mp_result = self.mp_face.detect(frame)
        if mp_result is None:
            return None, None, None

        face_box, eye_boxes, eye_centers, landmarks_35 = mp_result
        landmarks_points = np.c_[landmarks_35, np.ones((landmarks_35.shape[0], 1))].T

        square_box = _square_face_box(face_box, frame.shape)
        if square_box is None:
            return None, None, None

        face = self.get_crop_image(frame, square_box)
        if face is None:
            return None, None, None

        face = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE))
        img_tensor = self.transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            gaze = self.model({"face": img_tensor})
        gaze = gaze.detach().cpu().numpy().reshape(-1)

        yaw = float(gaze[0]) - YAW_BIAS
        pitch = float(gaze[1]) - PITCH_BIAS
        gaze_vector = normalize_vec(gaze_to_3d(yaw, pitch))
        gaze_vector[1] *= -1.0
        gaze_vector = util.MedianFilter(self.queue_gaze, gaze_vector)

        relative_eye_boxes, relative_eye_centers = to_relative_coordinates(
            eye_boxes, eye_centers, square_box
        )

        head_box = np.array([square_box[0], square_box[1]])
        right_eye_box = np.array([relative_eye_boxes[0][0], relative_eye_boxes[0][1]])
        left_eye_box = np.array([relative_eye_boxes[1][0], relative_eye_boxes[1][1]])

        pnp_tvec, pnp_R = self.mp_face.get_pose(frame)
        pnp_success = False
        pnp_distance = None

        if pnp_tvec is not None and pnp_R is not None:
            if (np.abs(pnp_tvec) > 10000).any():
                pnp_tvec = None
                pnp_R = None
            else:
                pnp_success = True
                pnp_distance = pnp_tvec[2, 0] / 10.0

        if pnp_R is not None:
            yaw_r, pitch_r, roll_r = util.rot_matrix_to_ypr(pnp_R)
            head_pose_angles = np.degrees([yaw_r, pitch_r, roll_r])
        else:
            head_pose_angles = np.zeros(3, dtype=np.float32)

        eye_info = {
            "gaze": gaze_vector,
            "EyeRLCenterPos": np.array(relative_eye_centers).reshape(-1),
            "HeadPosAnglesYPR": head_pose_angles,
            "HeadPosInFrame": head_box,
            "right_eye_box": right_eye_box,
            "left_eye_box": left_eye_box,
            "EyeState": [1, 1],
        }

        pnp_info = {
            "pnp_tvec": pnp_tvec,
            "pnp_R": pnp_R,
            "pnp_distance": pnp_distance,
            "pnp_success": pnp_success,
        }

        self.draw_eye_line(
            frame,
            face_box,
            eye_boxes,
            eye_centers,
            gaze_vector[0],
            gaze_vector[1],
        )
        if imshow:
            cv2.imshow("image", frame)

        return eye_info, landmarks_points, pnp_info

    def get_FaceFeatures(self, frame, imshow=False, face_boxes=None):
        mp_result = self.mp_face.detect(frame)
        if mp_result is None:
            return np.zeros((3, 35))

        _, _, _, landmarks_35 = mp_result
        return np.c_[landmarks_35, np.ones((landmarks_35.shape[0], 1))].T
