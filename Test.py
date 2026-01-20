import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math


# 禁用安全保护
pyautogui.FAILSAFE = False  

# 摄像头显示开关
camera_on = True  


# 手部模型初始化
mp_hands = mp.solutions.hands

# 绘图工具
mp_drawing = mp.solutions.drawing_utils

#获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 食指尖端索引
INDEX_FINGER_TIP = 8  




#鼠标控制参数
MOUSE_SCALING = 1.5     # 鼠标移动缩放因子
MOUSE_OFFSET_X = 50     # X轴偏移量
MOUSE_OFFSET_Y = 50     # Y轴偏移量
MOUSE_SMOOTHING = True  # 是否启用鼠标平滑
mouse_mode = False      # 鼠标控制模式开关


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




# 平滑滤波器（减少鼠标抖动）
class MouseSmoother:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x_buffer = []
        self.y_buffer = []
        
    def add_point(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
    
    def get_smoothed_point(self):
        if not self.x_buffer:
            return None
        return np.mean(self.x_buffer), np.mean(self.y_buffer)

# 鼠标平滑器实例
mouse_smoother = MouseSmoother(buffer_size=3)

#================================================================#

# 计算手指的角度：ab与bc之间的夹角
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

#手指状态匹配映射表
def fig_status(status):
    status_dict = {

    }
    return status_dict.get(status, "Unknown Pose")

#============================================================================#


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
        # 计算手指角度
        # 判断手指是否伸直：计算手指角度
        # 左手需要反向判断，因为坐标系镜像
        if handedness == "Right":
            angle = calculate_angle(tip, pip, mcp)
            if angle > 150:  # 如果角度接近180度（伸直），则认为手指伸直,阈值可以根据实际情况调整，通常150-180度可以认为是伸直
                fingers_up[i] = 2**i
        else:  # 左手
            angle = calculate_angle(tip, pip, mcp)
            if angle > 150:  # 角度阈值
                fingers_up[i] = 2**i
    
    # 2. 判断拇指（需要特殊处理）
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_ip = hand_landmarks.landmark[3]  # 拇指指间关节
    thumb_mcp = hand_landmarks.landmark[FINGER_MCP[0]]
    
    # 计算拇指的张开角度
    if handedness == "Right":
        # 右手
        if calculate_angle(thumb_tip, thumb_ip, thumb_mcp) > 155:
            fingers_up[0] = 1
    else:  # 左手
        if calculate_angle(thumb_tip, thumb_ip, thumb_mcp) > 155:
            fingers_up[0] = 1
    
    # 3. 计算手指系数
    detected_number = fig_status(sum(fingers_up))
    
    
    return fingers_up, detected_number

#============================================================================#
def control_mouse_with_index_finger(hand_landmarks, frame_width, frame_height, handedness="Right"):
    """
    使用食指指尖控制鼠标
    
    参数:
        hand_landmarks: MediaPipe检测到的手部关键点
        frame_width: 摄像头帧的宽度
        frame_height: 摄像头帧的高度
        handedness: 手性（"Left" 或 "Right"）
    """
    global mouse_mode, click_gesture_active
    
    # 获取食指尖端位置
    index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
    
    # 打印调试信息（可选）
    # print(f"食指原始坐标: x={index_tip.x:.3f}, y={index_tip.y:.3f}")
    
    # 将坐标转换为屏幕坐标
    # 注意：MediaPipe的坐标是归一化的 [0, 1]
    
    # 方法1：直接映射（最简单）
    screen_x = int((1 - index_tip.x) * screen_width * MOUSE_SCALING)  # 镜像翻转
    #screen_x = int(index_tip.x * screen_width * MOUSE_SCALING)
    screen_y = int((1 - index_tip.y) * screen_height * MOUSE_SCALING)
    
    # 方法2：如果摄像头画面是镜像的，可能需要翻转X坐标
    # 这取决于您是否在显示时使用了 cv2.flip(frame, 1)
    # screen_x = int((1 - index_tip.x) * screen_width * MOUSE_SCALING)  # 镜像翻转

    # BORDER_MARGIN = 50  # 距离屏幕边缘至少50像素
    
    # screen_x = max(BORDER_MARGIN, min(screen_x, screen_width - 1))
    # screen_y = max(BORDER_MARGIN, min(screen_y, screen_height - 1))


    
    # 边界检查（在缩放后进行）
    screen_x = max(0, min(screen_x, screen_width - 1))
    screen_y = max(0, min(screen_y, screen_height - 1))
    
    # 添加平滑处理
    if MOUSE_SMOOTHING:
        mouse_smoother.add_point(screen_x, screen_y)
        smoothed_point = mouse_smoother.get_smoothed_point()
        if smoothed_point:
            screen_x, screen_y = int(smoothed_point[0]), int(smoothed_point[1])
    
    # 防抖动：检查移动距离是否足够大
    if hasattr(control_mouse_with_index_finger, 'last_position'):
        last_x, last_y = control_mouse_with_index_finger.last_position
        distance = ((screen_x - last_x)**2 + (screen_y - last_y)**2)**0.5
        if distance < 5:  # 移动距离小于5像素时不更新，减少抖动
            return
    
    control_mouse_with_index_finger.last_position = (screen_x, screen_y)
    
    # 移动鼠标（添加平滑移动）
    try:
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    except Exception as e:
        print(f"鼠标移动错误类型: {type(e).__name__}")
        print(f"详细错误信息: {e}")
        print(f"尝试移动到的坐标: ({screen_x}, {screen_y})")
        print(f"屏幕尺寸: {screen_width}x{screen_height}")
        # 可以在这里重置鼠标位置
        # pyautogui.moveTo(screen_width // 2, screen_height // 2)


#================================================================#

# 检查鼠标控制手势（只有食指伸直）
def check_mouse_control_gesture(fingers_up):
    """
    检查是否只有食指伸直（鼠标控制手势）
    """
    # 只有食指伸直（食指=2，其他都是0）
    return (fingers_up[1] == 2 and 
            fingers_up[0] == 0 and 
            fingers_up[2] == 0 and 
            fingers_up[3] == 0 and 
            fingers_up[4] == 0)

#================================================================#

#  自定义绘制函数 - 关键点改为白框红圆
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
            
            cv2.line(frame, start_pos, end_pos, color, 2)
    
    # 3. 绘制所有关键点 - 统一为白框红圆
    # 首先绘制红色实心圆,食指指尖特殊处理（内部）
    for idx, landmark in enumerate(hand_landmarks.landmark):
        center = (int(landmark.x * w), int(landmark.y * h))
    
        if idx == 8:  # 食指指尖（索引8）
            cv2.circle(frame, center, 15, (255, 0, 0), -1)  # 蓝色实心圆，更大
        else:
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 红色实心圆
    
    # 然后绘制白色边框（外部）
    for landmark in hand_landmarks.landmark:
        center = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, center, 6, (255, 255, 255), 1)  # 白色边框，线宽1

#================================================================#




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
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取手性
            handedness = results.multi_handedness[i].classification[0].label
            
            # 统计手指状态
            fingers_up, detected_number = count_fingers(hand_landmarks, handedness)
            
            # 检查是否为鼠标控制手势
            if check_mouse_control_gesture(fingers_up):
                mouse_mode = True
                # 使用食指控制鼠标，传入frame尺寸
                control_mouse_with_index_finger(
                    hand_landmarks, 
                    frame.shape[1],  # 宽度
                    frame.shape[0],  # 高度
                    handedness
                )
        if camera_on:
            # 使用自定义绘制函数       
            for hand_landmarks in results.multi_hand_landmarks:
                 draw_custom_hand(frame, hand_landmarks)
            cv2.imshow('Colored Hand Tracking with Red Dots', frame)
    if cv2.waitKey(5) & 0xFF == 27:    # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()
