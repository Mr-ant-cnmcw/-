import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化手部模型 (参数可根据需要调整)
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

    # 如果检测到手，绘制关键点和连线
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow('MediaPipe Hands on Orange Pi', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()
