import cv2
import dlib
import numpy as np


# 方法：关键点编码：128D特征向量输出
def encoder_face(image, detector, predictor, encoder, upsample=1, jet=1):
    # 检测人脸
    faces = detector(image, upsample)
    # 对每张人脸进行关键点检测
    faces_keypoints = [predictor(image, face) for face in faces]  # 每张人脸的关键点
    return [np.array(encoder.compute_face_descriptor(image, face_keypoint, jet)) for face_keypoint in faces_keypoints]


# 方法：通过计算特征向量之间的欧氏距离，人脸比较判断是否为同一人，并输出对应的名称
def comapre_faces_in_order(face_encoding, test_encoding, names):
    distance = list(np.linalg.norm(np.array(face_encoding) - np.array(test_encoding), axis=1))
    return zip(*sorted(zip(distance, names)))


def main():
    # 读取图片（4+1张）
    image1 = cv2.imread("guo.jpg")
    image2 = cv2.imread("liu1.jpg")
    image3 = cv2.imread("liu2.jpg")
    image4 = cv2.imread("liu3.jpg")
    test_image = cv2.imread("liu4.jpg")
    image_names = ["guo,jpg", "liu1.jpg", "liu2.jpg", "liu3.jpg"]  # 存为名称数组

    # 加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 加载关键点的检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 加载人脸特征编码模型
    encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 方法：关键点编码：128D特征向量输出
    image1_128D = encoder_face(image1, detector, predictor, encoder)[0]
    image2_128D = encoder_face(image2, detector, predictor, encoder)[0]
    image3_128D = encoder_face(image3, detector, predictor, encoder)[0]
    image4_128D = encoder_face(image4, detector, predictor, encoder)[0]
    test_image_128D = encoder_face(test_image, detector, predictor, encoder)[0]
    images_list_128D = [image1_128D, image2_128D, image3_128D, image4_128D]  # 存为特征向量数组

    # 方法：通过计算特征向量之间的欧氏距离，人脸比较判断是否为同一人，并输出对应的名称
    distance, name = comapre_faces_in_order(images_list_128D, test_image_128D, image_names)
    print("distance: {}, \n names: {} ".format(distance, name))


if __name__ == '__main__':
    main()
