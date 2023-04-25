import cv2
import os

# 视频路径和图像路径
video_path = "/content/SPACE/data/video"
image_path = "/content/SPACE/data/images"

# 如果图像目录不存在，则创建
if not os.path.exists(image_path):
    os.makedirs(image_path)

# 遍历视频目录中的所有文件
for filename in os.listdir('/content/SPACE/data/video'):
    # 仅处理视频文件
    if not filename.endswith(".mpg"):
        continue

    # 打开视频文件
    video_file = os.path.join('/content/SPACE/data/video', filename)
    video_capture = cv2.VideoCapture(video_file)

    # 设置计数器
    count = 0

    # 循环读取视频帧并保存为图片
    while True:
        success, image = video_capture.read()
        if not success:
            break

        # 将帧保存为图像文件
        image_folder = os.path.join(image_path, os.path.splitext(filename)[0])

        
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        image_file = os.path.join(image_folder, "{:06d}.jpg".format(count))


        cv2.imwrite(image_file, image)

        count += 1

    # 释放视频捕捉器
    video_capture.release()