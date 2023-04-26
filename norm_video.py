import collections 
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import librosa 
import shutil
import os
import torchvision
from configparser import ConfigParser
from models import model
import cv2
import numpy as np
from models import funtion

audio_folder = '/content/drive/MyDrive/space_local/data/audio'
image_folder = '/content/drive/MyDrive/space_local/data/image'
# 获取音频文件名（不包含后缀）
audio_files = [os.path.splitext(f)[0] for f in os.listdir(audio_folder) if f.endswith('.wav')]

# 提取MFCC特征
# mfcc = audio_feature = funtion.auido_feature_extract(audio_path, int(self.sr), int(self.n_fft),int(self.n_mfcc), int(self.fps))


image_folder= '/content/drive/MyDrive/space_local/data/image'
image_counts = collections.Counter()



for folder in os.listdir(image_folder):
  folder_path= os.path.join(image_folder,folder)
  if os.path.isdir(folder_path):
    image_count= len(os.listdir(folder_path))#记录每个图片文件夹的所含图片数量
    image_counts[image_count] += 1
most_common_count = image_counts.most_common(1)[0][0]#记录出现最多次数


for folder in os.listdir(image_folder):#folder指的是小文件夹类似于bbae9n
    folder_path = os.path.join(image_folder, folder)
    if os.path.isdir(folder_path):
        image_count = len(os.listdir(folder_path))
        if image_count < most_common_count:
            # 删除数量不足的文件夹
            shutil.rmtree(folder_path)
            print(f"已删除文件夹 {folder_path}")


            audio_path = os.path.join(audio_folder, folder)
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除语音文件 {audio_path}")
        else:
            # 对数量足够的文件夹进行裁剪
            images = os.listdir(folder_path)
            for i in range(most_common_count, len(images)):
                image_path = os.path.join(folder_path, images[i])
                os.remove(image_path)
                print("已剪裁文件 {}".format(image_path))



# if mfcc.shape[1] < self.most_common_num_images:
#   raise ValueError(f"mfcc 帧数 ({mfcc.shape[0]}) 小于最常见的帧数 ({self.most_common_num_images})，跳过此样本。")
# mfcc = torch.from_numpy(mfcc).float()
# # 加载图像

# # 加载图像
# image_folder = os.path.join(self.image_folder, os.path.splitext(os.path.basename(audio_path))[0])
# image_files = os.listdir(image_folder)
# # 如果图像数目小于最常见次数，跳过该样本
# if len(image_files) < self.most_common_num_images:
#   raise ValueError("Not enough images for audio sample")
# # 如果图像数目大于最常见次数，裁减至最常见次数
# elif len(image_files) >= self.most_common_num_images:
#   image_files = image_files[:self.most_common_num_images]
