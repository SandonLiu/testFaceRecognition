import cv2
import face_recognition
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
def plot_landmarks_on_faces(image):
    # 灰度转换
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调用face_recognition库中的方法：face_landmarks()
    face_marks = face_recognition.face_landmarks(grey_image, None, "large")

    for landmarks_dict in face_marks:
        for landmarks_key in landmarks_dict.keys():
            for point in landmarks_dict[landmarks_key]:
                cv2.circle(image, point, 2, (255, 255, 255), -1)
    return image


def main():
    # 读取图片
    image = cv2.imread("family.jpg")

    # 方法：给人脸画框
    processed_image = plot_landmarks_on_faces(image.copy())

    # 方法：显示图片
    show_image(processed_image)


if __name__ == '__main__':
    main()
