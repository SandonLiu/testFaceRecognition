import cv2
import dlib
import matplotlib.pyplot as plt


# 方法：显示图片
def show_image(processed_image):
    # 创建画布,显示检测效果
    plt.figure(figsize=(9, 6))
    plt.suptitle("face landmarks by dlib", fontsize=14, fontweight="bold")
    plt.title("face landmarks")
    plt.axis("off")
    image_BGR = processed_image[:, :, ::-1]  # BRG to RGB
    plt.imshow(image_BGR)
    plt.show()


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


def main():
    # 读取图片
    image = cv2.imread("Tom.jpeg")

    # 方法：给人脸画框
    processed_image = plot_info_on_faces(image.copy())

    # 方法：显示图片
    show_image(processed_image)


if __name__ == '__main__':
    main()
