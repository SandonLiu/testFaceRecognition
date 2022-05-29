import cv2
import argparse

# 添加参数
parser = argparse.ArgumentParser()
parser.add_argument("vedio_path", help="视频文件路径")
args = parser.parse_args()
argsByVars = vars(args)

# 5 加载视频文件
capture = cv2.VideoCapture(argsByVars["vedio_path"])

# 6 读取视频
ret, frame = capture.read()  # ret:是否读取到了帧
while ret:
    cv2.imshow("video", frame)
    ret, frame = capture.read()  # 继续读取帧
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 7 释放资源
capture.release()
# 8 关闭窗口
cv2.destroyAllWindows()

# python read_video.py G:\AI\jupyter\videos\view.mp4