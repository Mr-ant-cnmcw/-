import cv2
import mediapipe as mp
import math

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

# 手指指尖索引（用于判断手指是否伸直）
FINGER_TIPS = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖索引
FINGER_PIPS = [2, 6, 10, 14, 18]  # 对应手指的第二个关节（作为弯曲判断参考点）
FINGER_MCP = [1, 5, 9, 13, 17]    # 对应手指的掌指关节





# 计算手指的角度：指尖-第二关节-掌根三个点的夹角
def calculate_angle(a, b, c):
    """
    计算三个点之间的角度（b是顶点）
    a: 指尖
    b: 第二关节
    c: 掌根
    """
    
    # 计算向量
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    
    # 计算点积
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    
    # 计算模长
    magnitude_ba = (ba[0]**2 + ba[1]**2) ** 0.5
    magnitude_bc = (bc[0]**2 + bc[1]**2) ** 0.5
    
    # 计算夹角（弧度）
    if magnitude_ba * magnitude_bc == 0:
        return 0
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(-1, min(1, cos_angle))  # 确保在[-1, 1]范围内
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

# 计算手指角度


def count_fingers(hand_landmarks, handedness="Right"):
    """
    统计伸直的手指数量并判断数字手势
    返回: (伸直手指数量, 识别的数字)
    """
    h, w, _ = frame.shape if 'frame' in locals() else (480, 640, 3)
    
    # 获取手腕位置（参考点）
    wrist = hand_landmarks.landmark[0]
    
    fingers_up = [0, 0, 0, 0, 0]  # 拇指、食指、中指、无名指、小指
    
    # 1. 判断除拇指外的四个手指（食指到小指）
    for i in range(1, 5):
        tip_idx = FINGER_TIPS[i]
        pip_idx = FINGER_PIPS[i]
        mcp_idx = FINGER_MCP[i]
        
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]



        angle = calculate_angle(tip, pip, mcp)
        
        # 判断手指是否伸直：计算手指角度
        # 左手需要反向判断，因为坐标系镜像
        if handedness == "Right":
            angle = calculate_angle(tip, pip, mcp)
            if angle > 150:  # 如果角度接近180度（伸直），则认为手指伸直,阈值可以根据实际情况调整，通常150-180度可以认为是伸直
                fingers_up[i] = 1
        else:  # 左手
            angle = calculate_angle(tip, pip, mcp)
            if angle > 150:  # 角度阈值
                fingers_up[i] = 1
    
    # 2. 判断拇指（需要特殊处理）
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_ip = hand_landmarks.landmark[3]  # 拇指指间关节
    thumb_mcp = hand_landmarks.landmark[FINGER_MCP[0]]
    
    # 计算拇指的张开角度
    if handedness == "Right":
        # 右手
        if calculate_angle(thumb_tip, thumb_ip, thumb_mcp) > 150:
            fingers_up[0] = 1
    else:  # 左手
        if calculate_angle(thumb_tip, thumb_ip, thumb_mcp) > 150:
            fingers_up[0] = 1
    
    # 3. 计算伸直的手指总数
    total_fingers = sum(fingers_up)
    
    # 4. 特殊手势识别（数字0-5）
    detected_number = total_fingers  # 默认对应数字
    
    # 数字0：握拳（没有手指伸直，但手指都弯曲）
    if total_fingers == 0:
        # 检查是否真的是握拳（不是其他手势）
        all_fingers_bent = True
        for i in range(5):
            tip = hand_landmarks.landmark[FINGER_TIPS[i]]
            pip = hand_landmarks.landmark[FINGER_PIPS[i]]
            if tip.y < pip.y:  # 如果有手指伸直
                all_fingers_bent = False
                break
        if all_fingers_bent:
            detected_number = 0
    
    # 数字5：五指张开（特殊检查）
    elif total_fingers == 5:
        # 确保拇指确实张开（不是误判）
        thumb_angle = math.atan2(thumb_tip.y - thumb_mcp.y, thumb_tip.x - thumb_mcp.x)
        if abs(thumb_angle) > 0.3:  # 拇指张开角度足够大
            detected_number = 5
        else:
            detected_number = 4  # 可能是4
    
    # 数字1-4：直接对应伸直手指数
    else:
        detected_number = total_fingers
    
    return fingers_up, detected_number



# 2. 自定义绘制函数 - 关键点改为白框红圆
def draw_custom_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    
    # 绘制手掌基部连接 (白色，稍细)
    for start_idx, end_idx in PALM_CONNECTIONS:
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        
        start_pos = (int(start_point.x * w), int(start_point.y * h))
        end_pos = (int(end_point.x * w), int(end_point.y * h))
        
        cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
    
    # 绘制每根手指的连线（保持不同颜色）
    for finger_indices, color in FINGER_CONNECTIONS:
        for i in range(len(finger_indices) - 1):
            start_idx = finger_indices[i]
            end_idx = finger_indices[i + 1]
            
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            
            start_pos = (int(start_point.x * w), int(start_point.y * h))
            end_pos = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start_pos, end_pos, color, 2)# 绘制手指连线

    
    # 3. 绘制所有关键点 - 统一为白框红圆
    # 首先绘制红色实心圆（内部）
    for landmark in hand_landmarks.landmark:
        center = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 红色实心圆
    
    # 然后绘制白色边框（外部）
    for landmark in hand_landmarks.landmark:
        center = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, center, 6, (255, 255, 255), 1)  # 白色边框，线宽1


# 如果有检测到的数字，显示在手腕上方
    if detected_number is not None:
        wrist = hand_landmarks.landmark[0]
        text_pos = (int(wrist.x * w) - 40, int(wrist.y * h) - 60)
        
        # 创建半透明背景
        text = f"Number: {detected_number}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        thickness = 2
        
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 绘制半透明矩形背景
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (text_pos[0] - 10, text_pos[1] - text_height - 10),
                     (text_pos[0] + text_width + 10, text_pos[1] + 10),
                     (0, 0, 0), -1)
        
        # 透明度混合
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制文本
        cv2.putText(frame, text, text_pos, font, font_scale, (0, 255, 255), thickness)







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

    # 如果检测到手:
    if results.multi_hand_landmarks:
      for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取手部类型（左/右）
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label
            
            # 统计手指并识别数字
            fingers_status, detected_number = count_fingers(hand_landmarks, handedness)
            
            # 在图像左上角显示手指状态
            status_text = f"Fingers: {sum(fingers_status)} ["
            for i, status in enumerate(fingers_status):
                status_text += 'UP' if status else 'DOWN'
                if i < 4:
                    status_text += ' '
            status_text += ']'
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 绘制手部标记和数字
            draw_custom_hand(frame, hand_landmarks)



    cv2.imshow('Colored Hand Tracking with Red Dots', frame)
    if cv2.waitKey(5) & 0xFF == 27:    # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()
