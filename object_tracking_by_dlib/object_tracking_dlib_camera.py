import cv2
import dlib

# 全局变量：存放鼠标事件的坐标点
points = []


# 方法：显示信息
def show_info(frame, tracking_state):
    pos1 = (10, 20)
    pos2 = (10, 40)
    pos3 = (10, 60)

    info1 = "put left button, select an area, starct tracking"
    info2 = " '1' : starct tracking ,  '2' : stop tacking , 'q' : exit "
    cv2.putText(frame, info1, pos1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, info2, pos2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    if tracking_state:
        cv2.putText(frame, "tracking now ...", pos3, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "stop tracking ...", pos3, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))


# 方法：鼠标点击事件
def mouse_event_handler(event, x, y, flags, parms):
    global points  # 全局调用
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
        points.append((x, y))


# 方法：将鼠标事件绑定到窗口上
def mouse_event_bind_to_window(nameWindow):
    cv2.namedWindow(nameWindow)
    cv2.setMouseCallback(nameWindow, mouse_event_handler)
    return nameWindow


# 主函数
def main():
    global points  # 全局调用

    # 读取摄像头
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("摄像头开启失败")

    # 设置窗口名称
    nameWindow = "Object Tracking"

    # 鼠标点击与窗口绑定
    mouse_event_bind_to_window(nameWindow)

    # 启动跟踪器 dlib.correlation_tracker()
    tracker = dlib.correlation_tracker()

    # 跟踪状态初始设为False
    tracking_state = False
    while True:
        # 通过摄像头，捕获-当前帧
        ret, frame = capture.read()  # ret:Boolean是否读取到了帧

        # 方法：显示提示信息
        show_info(frame, tracking_state)

        # 跟踪状态判定
        if not tracking_state:
            # 初次绘制出矩形框
            if len(points) == 2:
                # 绘制矩形框
                cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 2)
        else:
            # 更新跟踪，获取位置，绘制矩形框
            # 更新画面
            tracker.update(frame)
            # 获取物体位置坐标
            position = tracker.get_position()
            # 绘制新的矩形框
            cv2.rectangle(frame, (int(position.left()), int(position.top())),
                          (int(position.right()), int(position.bottom())), (0, 255, 0), 3)

        # 鼠标事件判断，根据按键：'1', '2', 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            if len(points) == 2:
                # dlib记录坐标位置，并启动跟踪器跟踪
                tracker.start_track(frame, dlib.rectangle(points[0][0], points[0][1], points[1][0], points[1][1]))
                tracking_state = True
                points = []
        if key == ord('2'):
            tracking_state = False
            points = []
        if key == ord('q'):
            break

        # 显示整体效果
        cv2.imshow(nameWindow, frame)

    # 释放资源
    capture.release()
    # 关闭窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# https://github.com/davisking/dlib-models
