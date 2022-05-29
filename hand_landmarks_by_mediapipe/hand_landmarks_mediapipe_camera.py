import cv2
import mediapipe as mp
import time

pre_Time = 0


# 方法：显示帧率
def show_fps(frame):
    global pre_Time  # 全局变量
    cur_Time = time.time()
    fps = 1 / (cur_Time - pre_Time)
    pre_Time = cur_Time
    cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  # 颜色，字号


# 方法：判断是否关闭摄像头
def close_camera_or_not(ret):
    # 未读取到帧，关闭摄像头
    if not ret:
        print("摄像头没有读取到画面，请检查摄像头")
        raise
    # 键盘输入“q",关闭摄像头
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q'键已按下，程序正常退出")
        raise


# 方法：landmarks信息输出
def output_landmarks_info(frame, hand_landmarks, output_flg):
    if not output_flg:
        return

    for i, lm in enumerate(hand_landmarks.landmark):
        x_Pos = int(lm.x * frame.shape[1])
        y_Pos = int(lm.y * frame.shape[0])
    cv2.putText(frame, str(i), (x_Pos - 25, y_Pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if i == 4:
        cv2.circle(frame, (x_Pos, y_Pos), 20, (166, 56, 56), cv2.FILLED)
    print(i, x_Pos, y_Pos)


# 主函数
def main():
    # 读取摄像头
    capture = cv2.VideoCapture(0)

    # 手部位置计算器初始设置
    hands_location_calculation = mp.solutions.hands
    hands_location = hands_location_calculation.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 绘制的相关参数初始设置
    draw = mp.solutions.drawing_utils
    hand_joint = draw.DrawingSpec(color=(255, 255, 255), thickness=3)  # 关节点
    hand_line = draw.DrawingSpec(color=(230, 243, 254), thickness=5)  # 连线

    try:
        while capture.isOpened():
            # 通过摄像头，捕获-当前帧
            ret, frame = capture.read()  # ret:Boolean是否读取到了帧

            # 方法：判断是否关闭摄像头
            close_camera_or_not(ret)

            # 绘制手部
            # BGR->RGB
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 手部位置计算
            hands = hands_location.process(frame_RGB)

            # 手部位置存在时
            if hands.multi_hand_landmarks:

                # 绘制
                for hand_landmarks in hands.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_landmarks, hands_location_calculation.HAND_CONNECTIONS,
                                        hand_joint, hand_line)
                    # 是否输出landmarks信息
                    output_landmarks_info(frame, hand_landmarks, True)

            # 方法：显示帧率
            show_fps(frame)

            # 显示当前帧
            cv2.imshow('hand_landmarks', frame)

    finally:
        # 释放资源
        capture.release()
        # 关闭窗口
        cv2.destroyAllWindows()


# 主程序入口
if __name__ == '__main__':
    main()
