import torch.nn as nn
import torch
import dataloader
from torch.utils.data import Dataset, DataLoader
import torchvision
from models import model
from models import funtion
import cv2
import numpy as np
from synergy3DMM import SynergyNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from configparser import ConfigParser
from itertools import zip_longest
# 如果有多片gpu时使用
# device_ids = [0, 1, 2]  # 多个GPU的ID
# model = MyModel()  # 自定义模型
# model = nn.DataParallel(model, device_ids=device_ids)

# 对于一片gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 读取config文件
config = ConfigParser()
config.read('/content/SPACE/config.ini')
audio_length,_ = config.get('S2L_LSTM', 'audio_length').split(';', 1)
lmk_dim,_ = config.get('S2L_LSTM', 'lmk_dim').split(';', 1)
#####定义模型########
audio_net = model.AudioEncoder().to(device)
lmk_net = model.Encoder68().to(device)
s2l_lstm = model.S2L_LSTMModel(int(audio_length)+int(lmk_dim), 1024, 2, 3).to(device)
params = []
params += list(audio_net.parameters())
params += list(lmk_net.parameters())
params += list(s2l_lstm.parameters())
# 创建Adam优化器实例
optimizer = optim.Adam(params, lr=1e-5)
# 创建学习率调度器实例
# scheduler = lr_scheduler.LinearLR(optimizer, 10000, 5e-4)
#########################
# 初始化h0,c0
h0 = torch.randn(2, 68, 1024).to(device)
c0 = torch.randn(2, 68, 1024).to(device)
#提取脸部标志所需模型
synergynet = SynergyNet()
#加载数据 dataloader  
audio_folder = "data/audio"
image_folder = "data/images"
dataset = dataloader.AudioImageDataset(audio_folder, image_folder, transform=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=torch.utils.data._utils.collate.default_collate)

#绝对误差损失函数
loss1 = nn.L1Loss()



for mfcc, images in dataloader:
  
  """
  mcff[batchsize,帧数，channels,系数]
  images[batchsize,帧数，H,W,C]
  """
  images = images.squeeze(0) # [帧数，H,W,C] 
  images = np.array(images)
  lmklist = []
  # 提取图像脸部标志点
  for image in images:
    lmk68 = funtion.get_landmarks(synergynet,image)[0]
    lmk68 = np.transpose(lmk68, (1, 0))
    lmklist.append(lmk68)
  # 去除第一帧，作为预测帧
  lmk_step = lmklist[1:]
  lmk_step = torch.Tensor(lmk_step).to(device)

  lmklist = np.array(lmklist) # [帧数，68个点，三维坐标]
  lmklist = torch.tensor(lmklist)
  lmklist = lmklist.type(torch.float32)
  lmklist = lmklist.to(device)   #投入GPU
  print(lmklist.shape)
  
  fps_length = lmklist.shape[0]   #帧数

  mfcc_cut = mfcc[-1,:fps_length,:,:]  # 帧数对齐
  mfcc_cut = mfcc_cut.to(device)
  print(mfcc_cut.shape)
  audio_feature = audio_net(mfcc_cut) #[帧数，C，length]  语音编码器
  audio_feature = audio_feature.reshape(fps_length,1,-1) #[帧数，1,feature_length]
  print(audio_feature.shape)
  count = 0
  for audio_fps, image_fps, next_step in zip_longest(audio_feature, lmklist, lmk_step, fillvalue=None):
    """
    audio_fps:[1,length]
    image_fps:[68,3]
    """
    if next_step is None:
      print(11111)
      break
    image_code = lmk_net(image_fps)  #经过脸部标志编码器 [68,output]
    feature_add = funtion.lm_splice(audio_fps, image_code) #[1,68,1027]
    predict,h0,c0 = s2l_lstm(feature_add, h0,c0)
    loss = loss1(predict,next_step)
    optimizer.step()
    audio_net.zero_grad()
    lmk_net.zero_grad()
    s2l_lstm.zero_grad()
    print(loss)
    
  
  # print(mfcc.shape,images.shape)