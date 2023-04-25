import numpy as np
import cv2
import torch
from torch import nn
import landmarks2
from models import model
from configparser import ConfigParser
import mcff
import librosa
from synergy3DMM import SynergyNet

config = ConfigParser()
# 读取config文件
config.read('/content/SPACE/config.ini')
sr = config.get('mcff', 'sr')
n_fft = config.get('mcff', 'n_fft')
n_mfcc = config.get('mcff', 'n_mfcc')
fps = config.get('mcff', 'fps')
audio_feature = model.auido_feature_extract('/content/SPACE/data/audio/text.wav', int(sr), int(n_fft),
                                            int(n_mfcc), int(fps))

ad = model.AudioEncoder()
mcff = torch.from_numpy(audio_feature)#语音信息要用torch.from_numpy()转化为tensor
audio_feature = ad(mcff)
print('mcff.shape',audio_feature.shape)
audio_feature = audio_feature.reshape(601,1,-1)
print('mfcc.shape',audio_feature.shape) 
# audio_feature是经过网络的语音信息



# 提取图像
synergynet = SynergyNet()
image = cv2.imread('/content/SPACE/SynergyNet/img/sample_3.jpg')
lmk_old= model.get_landmarks(synergynet,image)[0]#没通过网络的
print('lmk_old.shpae',lmk_old.shape)
lmk_old= torch.from_numpy(lmk_old)
# 通过网络
# 在模型之前将输入和权重张量的数据类型都转换为float32
ec68=model.Encoder68()
lmk_old = lmk_old.type(torch.float32)
print('lmk_old',lmk_old.shape)
lmk_old= lmk_old.reshape(68,-1)
lmk_net = ec68(lmk_old)
# lmk_net = lmk_net.type(torch.float32)
print('lmk_net.shape',lmk_net.shape) 

# 定义lstm网络
s2l_lstm = model.S2L_LSTMModel(audio_feature.shape[-1]+lmk_net.shape[-1], 1024, 2, 3)
# 初始化h0,c0
h0 = torch.randn(2, 68, 1024)
c0 = torch.randn(2, 68, 1024)
#TOD0 dataiter
for audio_feature_per in audio_feature: #[601,1,1024]

  add = model.lm_splice(audio_feature_per, lmk_net)
  predict,_,_ = s2l_lstm(add, h0,c0)
  print(add.shape,predict.shape)



