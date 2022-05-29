import cv2
import argparse

# 定义参数
parser = argparse.ArgumentParser()  # 获取所有参数
parser.add_argument("path_image", help="图片路径")  # 添加参数
args = parser.parse_args()  # 解析所有参数

# 显示图片，方式1
img1 = cv2.imread(args.path_image)  # 加载图片
cv2.imshow("img1", img1)

# 显示图片，方式2
args_dict = vars(parser.parse_args())  # {"path_image" : "image/test1.jpg"}
img2 = cv2.imread(args_dict["path_image"])
cv2.imshow("img2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# python main.py G:\AI\jupyter\images\test1.jpg
