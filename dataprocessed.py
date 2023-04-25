import os
import cv2
import shutil

# 视频文件夹路径
video_dir = 'data/video'

# 音频文件夹路径
audio_dir = 'data/audio'

# 输出文件夹路径
output_dir = 'datapro'

# 创建用于保存图片的文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历视频文件夹中的所有视频
for video_filename in os.listdir(video_dir):
    # 确保文件是视频文件
    if not video_filename.endswith('.mpg'):
        continue

    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(video_filename)[0]

    # 创建用于保存视频帧的文件夹
    frames_dir = f'{output_dir}/{video_name}'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 使用OpenCV读取视频
    video_path = f'{video_dir}/{video_filename}'
    cap = cv2.VideoCapture(video_path)

    # 逐帧提取视频中的图片并保存
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_path = f'{frames_dir}/{frame_index:05d}.png'
        cv2.imwrite(output_path, frame)
        frame_index += 1

    # 释放视频资源
    cap.release()

    # 查找与视频文件对应的音频文件
    audio_filename = f'{video_name}.wav'
    audio_path = f'{audio_dir}/{audio_filename}'
    if not os.path.exists(audio_path):
        continue

    # 复制音频文件到输出文件夹
    output_audio_path = f'{output_dir}/{audio_filename}'
    shutil.copyfile(audio_path, output_audio_path)