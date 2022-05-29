import cv2
import dlib
import numpy
import matplotlib.pyplot as plt


# 方法：显示图片
def show_image(processed_image):
    # 创建画布,显示检测效果
    plt.figure(figsize=(9, 6))
    plt.suptitle("face detection by dlib", fontsize=14, fontweight="bold")
    plt.title("face detection")
    plt.axis("off")
    image_BGR = processed_image[:, :, ::-1]  # BRG to RGB
    plt.imshow(image_BGR)
    plt.show()


# 方法：画框（给人脸）
def plot_rectangle_on_faces(image):
    # 1.识别人脸
    # 灰度转换
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 调用dlib库中的检测器
    detector = dlib.get_frontal_face_detector()
    detected_faces = detector(grey_image, 1)  # 1表示将图片放大一倍

    # 2.给人脸画框
    # 拿到检测到的人脸数据，返回4个值，坐标（x,y)，矩形框的宽高width,height
    for face in detected_faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)  # 对角线， 颜色，矩形框宽px
    return image


def main():
    # 读取图片
    image = cv2.imread("family.jpg")

    # 方法：给人脸画框
    processed_image = plot_rectangle_on_faces(image.copy())

    # 方法：显示图片
    show_image(processed_image)


if __name__ == '__main__':
    main()
