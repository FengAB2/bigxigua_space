import torch.nn as nn
import torch
import dataloader
from torch.types import Device
from torch.utils.data import Dataset, DataLoader
import torchvision
from models import model, funtion
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
hidden_size = int(config.get('S2L_LSTM', 'hidden_size'))
batchsize = int(config.get('train', 'batchsize'))
epoch = int(config.get('train', 'epoch'))
pretrain = config.get('train', 'pretrain')

#####定义模型########
audio_net = model.AudioEncoder().to(device)
lmk_net = model.Encoder68().to(device)
s2l_lstm = model.S2L_LSTMModel(int(audio_length)+int(lmk_dim)*68, hidden_size, 2, 68*3).to(device)
if pretrain == True:
  nets_loaded = torch.load('models1.pth')
  audio_net.load_state_dict(nets_loaded['audio_net'])
  lmk_net.load_state_dict(nets_loaded['lmk_net'])
  s2l_lstm.load_state_dict(nets_loaded['s2l_lstm'])
params = []
params += list(audio_net.parameters())
params += list(lmk_net.parameters())
params += list(s2l_lstm.parameters())
# 创建Adam优化器实例
optimizer = optim.Adam(params, lr=1e-5)
# 创建学习率调度器实例
# scheduler = lr_scheduler.LinearLR(optimizer, 10000, 5e-4)
#########################

#提取脸部标志所需模型
synergynet = SynergyNet()
#加载数据 dataloader  
audio_folder = "data/audio"
image_folder = "data/images"
dataset = dataloader.AudioImageDataset(audio_folder, image_folder, transform=None)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

#绝对误差损失函数
loss1 = nn.L1Loss()
step = 0
for i in range(epoch):
  print('----------------第{}轮训练开始-------------------'.format(i+1))
  print("dataloader读取")
  for mfccs, images in dataloader:
    # 初始化h0,c0
    h0 = torch.randn(2, batchsize, 1024).to(device)
    c0 = torch.randn(2, batchsize, 1024).to(device)
    # print("11111")
    """
    mcffs[batchsize,帧数，channels,系数]
    images[batchsize,帧数，H,W,C]
    都是tensor类型
    """ 
    images = np.array(images)
    lmklist,lmk_step = funtion.extract_lankmarks(images,synergynet)
    # print(lmklist.shape,lmk_step.shape,111)
    """
    lmklist:[batchsize,fps,points,dim] 正常帧
    lmk_step:[batchsize,fps,points-1,dim] 少了第一帧
    """
    fps_length = lmklist.shape[1]   #帧数

    #将每一帧的音频信息送进网络处理
    mfccs = mfccs.to(device)
    processed_outputs = []
    for mfcc in mfccs:
      #[fps,c,length]
      audio_feature = audio_net(mfcc)
      # print(audio_feature.shape)
      processed_outputs.append(audio_feature)
    # 将处理后的结果拼接成一个batch [batch,fps,c,len]
    batch_mfcc = torch.stack(processed_outputs, dim=0)
    # print(batch_mfcc.shape,222)

    mfcc_cut = batch_mfcc[:,:fps_length,:,:]  # 帧数对齐
    mfcc_cut = mfcc_cut.to(device)
    audio_feature = mfcc_cut.reshape(2,fps_length,1,-1) #[batch，帧数，1,feature_length]]

    #对数据进行维度转换[fps,batch......]
    audio_feature = audio_feature.transpose(1,0)
    lmklist = lmklist.transpose(1,0)
    lmk_step = lmk_step.transpose(1,0)
    # print(audio_feature.shape)
    # print(lmklist.shape)
    # print(lmk_step.shape)
    
    #将数据送入GPU
    lmklist = lmklist.to(device)
    lmk_step = lmk_step.to(device)
    
    """
    lmklist:[fps,batchsize,points,dim] 正常帧
    lmk_step:[fps-1,batchsize,points,dim] 少了第一帧
    audio_feature:[fps，batchsize，1,feature_length]
    """
    ############################################
    
    count = 0
    loss_all = 0
    for audio_fps, image_fps, next_step in zip_longest(audio_feature, lmklist, lmk_step, fillvalue=None):
      """
      audio_fps:[batch，1,length]
      image_fps:[batch，68,3]
      next_step:[batch,68,3]
      """
      if next_step is None:
        print(11111)
        break
      image_code = lmk_net(image_fps)  #经过脸部标志编码器 [batch,204]
      # print(image_code.shape,111) #[2,204]
      feature_add = funtion.lm_splice(audio_fps, image_code) #[T,batchsize,length]
      # print(feature_add.shape,3333)
      predict,h0,c0 = s2l_lstm(feature_add,h0,c0)
      # print(predict.shape)
      predict = predict.reshape(batchsize,68,3)
      ##只对循环里面的网络梯度清零
      s2l_lstm.zero_grad()
      lmk_net.zero_grad()
      loss = loss1(predict,next_step)
      loss.backward(retain_graph=True)
      optimizer.step()
      # audio_net.zero_grad()
      # lmk_net.zero_grad()
      # s2l_lstm.zero_grad()
      loss_all += loss
      count += 1
      step += 1
      if step%5000==0:
        print('第{}次训练'.format(step))
        print('参数保存')
        # 保存模型参数
        state_dicts = {'audio_net': audio_net.state_dict(), 'lmk_net': lmk_net.state_dict(), 
        's2l_lstm': s2l_lstm.state_dict()}
        torch.save(state_dicts, 'models1.pth')
    print(loss_all/count)
    ##对所有网络梯度清零
    optimizer.zero_grad()
    
     
    
    # # print(mfcc.shape,images.shape)