import cv2
import argparse

# 添加参数
parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="哪个摄像头(从0开始，0，1，2...)", type=int)
args = parser.parse_args()
argsByVars = vars(args)

print("the camera index: ", argsByVars["index_camera"])

# 捕获摄像头视频
capture = cv2.VideoCapture(argsByVars["index_camera"])
# 判断摄像头是否开启
if (not capture.isOpened()):
    print("摄像头开启失败")

# 获取帧的宽,高,帧率
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
print("帧宽：{}".format(frame_width))
print("帧高：{}".format(frame_height))
print("FPS：{}".format(fps))

# 从摄像头读取视频，直到关闭
while capture.isOpened():
    # 通过摄像头，捕获当前帧
    ret, frame = capture.read()
    # 对当前帧灰度化处理
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 显示每一帧（视频流）
    cv2.imshow("OriginalSteam(Press q to exit)", frame)
    cv2.imshow("GreySteam(Press q to exit)", grey_frame)
    # 键盘输入“q",关闭摄像头
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 释放资源
capture.release()
# 关闭窗口
cv2.destroyAllWindows()

# 参数默认为0，反正我就一个摄像头，有俩选第二个的话，就写1
# python read_camera.py 0
