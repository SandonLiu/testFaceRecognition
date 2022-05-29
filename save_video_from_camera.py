import cv2
import argparse

# 添加参数
parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="哪个摄像头(从0开始，0，1，2...)", type=int)
parser.add_argument("video_outputPath", help="视频保存路径")
args = parser.parse_args()
argsByVars = vars(args)
print("the camera index: ", argsByVars["index_camera"])

# 捕获摄像头视频
capture = cv2.VideoCapture(argsByVars["index_camera"])
# 判断摄像头是否开启
if not capture.isOpened():
    print("摄像头开启失败")

# 获取帧的宽,高,帧率
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
print("帧宽：{}".format(frame_width))
print("帧高：{}".format(frame_height))
print("FPS：{}".format(fps))

# 对将要保存的视频进行编码
grey_flg = False  # True：灰度化处理
output_video = cv2.VideoWriter(argsByVars["video_outputPath"], cv2.VideoWriter_fourcc(*"XVID"), int(fps),
                               (int(frame_width), int(frame_height)), not grey_flg)


def grayScale(frame, grey_flg):
    if grey_flg:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        return frame


# 从摄像头读取视频，直到关闭
while capture.isOpened():
    # 通过摄像头，捕获-当前帧
    ret, frame = capture.read()  # ret:Boolean是否读取到了帧
    if ret:
        # 是否灰度化处理-当前帧
        processed_frame = grayScale(frame, grey_flg)
        # 写入视频文件-处理后的当前帧
        output_video.write(processed_frame)
        # 显示-当前帧（视频流）
        cv2.imshow("Original (Press q to exit)", frame)
        cv2.imshow("Processed (Press q to exit)", processed_frame)
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
capture.release()
# 关闭窗口
cv2.destroyAllWindows()

# python save_video_from_camera.py 0 G:\AI\jupyter\videos\videoOutputTest.mp4
