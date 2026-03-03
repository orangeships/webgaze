import cv2
import mediapipe as mp

class FaceMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True, # 必须开启以获取更精准的眼部边缘
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 50 个面部稳定点 
        self.stable_indices = [
            1, 2, 4, 5, 6, 8, 9, 10, 151, 168, 193, 197, 454,397, # 中轴线与边缘
            21, 71, 107, 336, 291, 334, 127,58, 93, 132, 361,102,331, # 额头与面颊
            103, 108, 109, 337, 338, 297, 298, 332, # 眉弓上方
            54, 67, 10, 284, 298, 337, # 更多额头点
            61, 291, 199, 211, 431, 264, 445, 200, 164, 397 # 嘴周稳定区域
        ]
        
        # 眼睛 8 点包围圈 (用于裁切)
        # 顺序：外眼角, 内眼角, 上缘(3点), 下缘(3点)
        self.l_eye_8 = [33, 133, 160, 159, 158, 144, 145, 153]
        self.r_eye_8 = [362, 263, 385, 386, 387, 373, 374, 380]

    def process_frame(self, frame):
        # 1. Resize 优化 (640w 兼顾速度与精度)
        h, w = frame.shape[:2]
        ratio = 640 / w
        input_w, input_h = 640, int(h * ratio)
        small_frame = cv2.resize(frame, (input_w, input_h))
        
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 绘制 50 个稳定点 (绿色)
                for idx in self.stable_indices:
                    lm = face_landmarks.landmark[idx]
                    pos = (int(lm.x * input_w), int(lm.y * input_h))
                    cv2.circle(small_frame, pos, 1, (0, 255, 0), -1)
                
                # 绘制左眼 8 点 (红色)
                for idx in self.l_eye_8:
                    lm = face_landmarks.landmark[idx]
                    pos = (int(lm.x * input_w), int(lm.y * input_h))
                    cv2.circle(small_frame, pos, 2, (0, 0, 255), -1)
                
                # 绘制右眼 8 点 (蓝色)
                for idx in self.r_eye_8:
                    lm = face_landmarks.landmark[idx]
                    pos = (int(lm.x * input_w), int(lm.y * input_h))
                    cv2.circle(small_frame, pos, 2, (255, 0, 0), -1)
                    
        return small_frame

if __name__ == "__main__":
    monitor = FaceMonitor()
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        out = monitor.process_frame(frame)
        cv2.imshow('Eye & Face Tracking', out)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()