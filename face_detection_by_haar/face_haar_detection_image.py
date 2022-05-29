import cv2
import numpy as np
import matplotlib.pyplot as plt


# 方法：显示图片
def show_image(image, title, position):
    # BGR to RGB
    img_RGB = image[:, :, ::-1]  # 高，宽不变，通道反向一下
    plt.subplot(2, 2, position)  # 显示4张图片（两行两列）
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")  # 不显示坐标轴


# 方法：对人脸画框
def plot_rectangle(image, faces):
    # 拿到检测到的人脸数据，返回4个值，坐标（x,y)，矩形框的宽高width,height
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # 对角线，蓝色，矩形框宽3px
    return image


# 主函数
def main():
    # 读取图片
    image = cv2.imread("girls.jpg")
    # 转换为灰度图片
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 通过OpenCV自带的方法cv2.CascadeClassifier()加载级联分类器
    face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # 对人脸进行检测
    face_alt2_detect = face_alt2.detectMultiScale(grey_image)

    # 【方法】：对人脸画框
    face_alt2_result = plot_rectangle(image.copy(), face_alt2_detect)

    # 创建画布
    plt.figure(figsize=(9, 6))
    plt.suptitle("use HaarCascade to detect face", fontsize=14, fontweight="bold")

    # 【方法】：显示图片(整体效果)
    show_image(face_alt2_result, "face_alt2", 1)
    plt.show()


# 主程序入口
if __name__ == '__main__':
    main()
