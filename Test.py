import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 1. 定义每根手指的连接关系和颜色（只用于连线）
FINGER_CONNECTIONS = [
    # 拇指 (蓝色)
    ([1, 2, 3, 4], (255, 0, 0)),     # 蓝
    # 食指 (绿色)
    ([5, 6, 7, 8], (0, 255, 0)),     # 绿
    # 中指 (红色)
    ([9, 10, 11, 12], (0, 0, 255)),  # 红
    # 无名指 (青色)
    ([13, 14, 15, 16], (255, 255, 0)), # 青
    # 小指 (紫色)
    ([17, 18, 19, 20], (255, 0, 255)) # 紫
]

# 手掌基部连接 (白色)
PALM_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17),
    (5, 9), (9, 13), (13, 17)
]

# 2. 自定义绘制函数 - 关键点改为白框红圆
def draw_custom_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    
    # 绘制手掌基部连接 (白色，稍细)
    for start_idx, end_idx in PALM_CONNECTIONS:
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        
        start_pos = (int(start_point.x * w), int(start_point.y * h))
        end_pos = (int(end_point.x * w), int(end_point.y * h))
        
        cv2.line(frame, start_pos, end_pos, (255, 255, 255), 1)
    
    # 绘制每根手指的连线（保持不同颜色）
    for finger_indices, color in FINGER_CONNECTIONS:
        for i in range(len(finger_indices) - 1):
            start_idx = finger_indices[i]
            end_idx = finger_indices[i + 1]
            
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            
            start_pos = (int(start_point.x * w), int(start_point.y * h))
            end_pos = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start_pos, end_pos, color, 1)
    
    # 3. 绘制所有关键点 - 统一为白框红圆
    # 首先绘制红色实心圆（内部）
    for landmark in hand_landmarks.landmark:
        center = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 红色实心圆
    
    # 然后绘制白色边框（外部）
    for landmark in hand_landmarks.landmark:
        center = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, center, 6, (255, 255, 255), 1)  # 白色边框，线宽1

# 初始化手部模型
hands = mp_hands.Hands(
    static_image_mode=False,      # 视频流模式
    max_num_hands=2,              # 最多检测几只手
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 打开摄像头
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # MediaPipe需要RGB图像
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 如果检测到手，使用自定义绘制函数
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_custom_hand(frame, hand_landmarks)

    cv2.imshow('Colored Hand Tracking with Red Dots', frame)
    if cv2.waitKey(5) & 0xFF == 27:    # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()
