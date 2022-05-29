import cv2
import face_recognition


# 输出摄像头信息
def printCameraInfo(capture):
    # 获取帧的宽,高,帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("帧宽：{}".format(frame_width))
    print("帧高：{}".format(frame_height))
    print("FPS：{}".format(fps))


# 方法：识别人脸并画框
def plot_landmarks_on_faces(image):
    # 灰度转换
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调用face_recognition库中的方法：face_landmarks()
    face_info_by_organs = face_recognition.face_landmarks(grey_image, None, "large")

    for organ_info_map in face_info_by_organs:
        for organ in organ_info_map.keys():
            for point in organ_info_map[organ]:
                cv2.circle(image, point, 1, (255, 255, 255), -1)
    return image


# 主函数
def main():
    # 读取摄像头
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("摄像头开启失败")
        return

    printCameraInfo(capture)

    while capture.isOpened():
        # 通过摄像头，捕获-当前帧
        ret, frame = capture.read()  # ret:Boolean是否读取到了帧

        # 未读取到帧，关闭摄像头
        if not ret:
            print("未读取到当前帧")
            break
        # 键盘输入“q",关闭摄像头
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 【方法】：识别人脸并画框
        processed_frame = plot_landmarks_on_faces(frame)

        # 显示当前帧
        cv2.imshow("face landmarks", processed_frame)

    # 释放资源
    capture.release()
    # 关闭窗口
    cv2.destroyAllWindows()


# 主程序入口
if __name__ == '__main__':
    main()
