import cv2


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
        # 未读取到帧，退出
        if not ret:
            print("未读取到当前帧")
            break

        # 【方法】：画框(对人脸)
        processed_frame = plot_rectangle_on_faces(frame)

        # 显示当前帧
        cv2.imshow("face detection", processed_frame)

        # 键盘输入“q",关闭摄像头
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    # 关闭窗口
    cv2.destroyAllWindows()


def printCameraInfo(capture):
    # 获取帧的宽,高,帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("帧宽：{}".format(frame_width))
    print("帧高：{}".format(frame_height))
    print("FPS：{}".format(fps))


# 方法：对人脸画框
def plot_rectangle_on_faces(image):
    # 转换为灰度图片
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 通过OpenCV自带的方法cv2.CascadeClassifier()加载级联分类器,对人脸进行检测
    face_detect_method = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    detected_faces_info = face_detect_method.detectMultiScale(grey_image)

    # 拿到检测到的人脸数据，返回4个值，坐标（x,y)，矩形框的宽高width,height
    for (x, y, w, h) in detected_faces_info:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 对角线， 颜色，矩形框宽px
    return image


# 主程序入口
if __name__ == '__main__':
    main()
