import cv2
import dlib


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
def plot_info_on_faces(image):
    # 1.识别人脸
    # 调用人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 灰度转换
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    detected_faces = detector(grey_image, 1)  # 1表示将图片放大一倍

    # 2.给人脸绘制信息
    for face in detected_faces:
        # 2.1 绘制矩形框
        # 拿到检测到的人脸数据，返回4个值，坐标（x,y)，矩形框的宽高width,height
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)  # 对角线， 颜色，矩形框宽px

        # 2.2 绘制关键点
        # 加载预测关键点模型
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 预测关键点
        shape = predictor(image, face)
        # 获取关键点坐标并绘图
        for pt in shape.parts():
            # 获取坐标
            pt_positon = (pt.x, pt.y)
            # 绘制关键点
            cv2.circle(image, pt_positon, 1, (255, 255, 255), -1)  # 坐标，半径px，颜色，实心空心

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
        processed_frame = plot_info_on_faces(frame)

        # 显示当前帧
        cv2.imshow("face detection", processed_frame)

    # 释放资源
    capture.release()
    # 关闭窗口
    cv2.destroyAllWindows()


# 主程序入口
if __name__ == '__main__':
    main()
