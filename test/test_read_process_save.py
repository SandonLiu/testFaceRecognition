import cv2
import argparse

# 添加参数
parser = argparse.ArgumentParser()
parser.add_argument("img_input_path", help="读取图片")
parser.add_argument("img_output_path", help="保存图片")
args = parser.parse_args()
argsByVars = vars(args)

# 读取图片
img_original = cv2.imread(argsByVars["img_input_path"])

# 处理图片
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# 保存图片
cv2.imwrite(argsByVars["img_output_path"], img_gray)

# 显示对比
cv2.imshow("img_original", img_original)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# python test_read_process_save.py G:\AI\jupyter\images\test1.jpg G:\AI\jupyter\images\test2.jpg