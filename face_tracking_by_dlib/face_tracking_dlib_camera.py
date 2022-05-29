import cv2
import dlib


# 方法：保存视频的参数设置
def set_output_video(capture, vedio_name):
    frame_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output = cv2.VideoWriter(vedio_name, cv2.VideoWriter_fourcc(*"XVID"), int(frame_fps),
                             (int(frame_width), int(frame_height)), True)
    return output


# 方法：显示提示信息
def show_info(frame, tracking_state):
    position1 = (20, 40)
    position2 = (20, 80)
    cv2.putText(frame, "'1' : reset ", position1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    # 根据跟踪状态，显示不同的信息
    if tracking_state:
        cv2.putText(frame, "tracking now ...", position2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    else:
        cv2.putText(frame, "no tracking ...", position2, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))


# 方法：人脸追踪
def tracking_face(detector, tractor, tracking_state, frame):
    # 如果没有跟踪， 启动跟踪器
    if not tracking_state:
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(grey_image, 1)
        # 检测到人脸数量>0
        if len(faces) > 0:
            tractor.start_track(frame, faces[0])
            tracking_state = True

    # 正在跟踪，实时获取人脸的位置，显示
    if tracking_state:
        tractor.update(frame)  # 更新画面
        position = tractor.get_position()  # 获取人脸的坐标
        cv2.rectangle(frame, (int(position.left()), int(position.top())),
                      (int(position.right()), int(position.bottom())), (0, 0, 255), 2)

    # 方法：显示提示信息
    show_info(frame, tracking_state)

    return (frame, tracking_state)


# 主函数
def main():
    # 读取摄像头
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("摄像头开启失败")
        return

    # 基于dlib库获取人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 基于dlib库实时跟踪
    tractor = dlib.correlation_tracker()
    # 设置输出视频的参数
    output = set_output_video(capture, "test_tracking.mp4")

    # 跟踪状态初始设为False
    tracking_state = False
    while True:
        # 通过摄像头，捕获-当前帧
        ret, frame = capture.read()  # ret:Boolean是否读取到了帧

        # 未读取到帧，关闭摄像头
        if not ret:
            print("未读取到当前帧")
            break
        # 键盘输入“q",关闭摄像头
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 方法：人脸追踪
        (processed_frame, tracking_state) = tracking_face(detector, tractor, tracking_state, frame)

        # 显示当前帧
        cv2.imshow("face tracking", processed_frame)

        # 保存视频
        output.write(processed_frame)

    # 释放资源
    capture.release()
    # 关闭窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
