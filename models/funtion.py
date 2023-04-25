import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
from configparser import ConfigParser

# landmark提取模型
# dim=0: 2d的lmk(3,68)
# dim=1: 3d的mesh(3, 53215)
# dim=2: 3d的lmk带有batch_size(1,3,68)
def get_landmarks(model,image):

  # get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
  lmk3d, mesh, pose = model.get_all_outputs(image)
  lmk3d= np.array(lmk3d)
  mesh= np.array(mesh)
  pose= np.array(pose)
  # print('lmk3d.shape:',lmk3d.shape)
  # print('mesh.shape:',mesh.shape)
  # print('pose.shape:',pose.shape)

  landmarks = lmk3d[0] # 提取每个面部特征点的三维坐标，并转置以适应下面的计算
  landmarks= landmarks.reshape(3,-1)
  # print('landmarks.shape',landmarks.shape)

  R,_ = cv2.Rodrigues(pose[0][0]) # 从旋转向量获取旋转矩阵
  t = pose[0][1]
  # print('t.shape',t.shape)

  # print('R.shape',R.shape)

  # print('mesh[0].shape',mesh[0].shape)



  rotated_lmk= np.dot(R,landmarks)
  rotated_lmk+=t[:,np.newaxis]
  # print('rotated_lmk.shape',rotated_lmk.shape)
  lmk_center= np.mean(rotated_lmk,axis=1)
  translated_lmk= rotated_lmk- lmk_center.reshape((-1,1))
  translated_3_lmk=translated_lmk[np.newaxis,:,:]

  translated_3_lmk= translated_3_lmk.reshape(1,68,3)



  rotated_mesh = np.dot(R, mesh[0]) # 旋转和平移每个顶点
  rotated_mesh+= t[:,np.newaxis]
  # print('rotated_mesh.shpae',rotated_mesh.shape)

  # 将模型中心点移动到原点
  center = np.mean(rotated_mesh, axis=1)
  # print('center.shape',center.shape)
  translated_mesh = rotated_mesh - center.reshape((-1, 1))
  

  return translated_lmk,translated_mesh,translated_3_lmk

# 提取MFCC特征
# 输出的mfcc是(601,1,40)

def auido_feature_extract(audio_path, sr, n_fft, n_mfcc,fps):
  """
  :param audio_path:the path to 'audio.wav'
  :param sr:采样率
  :param n_fft:FFT窗口大小
  :param n_mfcc:要提取的MFCC系数数量
  :param fps:音频帧数
  :return:mcff feature
  """
  # 加载音频文件
  y, sr = librosa.load(audio_path, sr=sr)  # 采样率为44100Hz,y表示音频信号
  hop_length = int(sr/fps)  #表示帧移，即每次移动的采样点数
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=int(sr/fps), n_mfcc=n_mfcc)
  # 将mfcc转化成(-1,1,40)添加一个通道维度
  mfcc= mfcc.reshape(-1,1,40)
  return mfcc

# 定义拼接lm_splice,
# 将经过音频编码器和人脸标志编码器的结果进行拼接，此处是拼接一帧

def lm_splice(mfcc_per,lmk_per): 
  """
    mfcc_per:[batch,1，1024*n]
    lmk_per:[batch,204]
  """
  
  batchsize = lmk_per.shape[0]
  lmk_per = lmk_per.reshape(batchsize,1,-1)
  print('1',lmk_per.shape)
  #print("Frame {}: {}".format(frame, mfcc_per))  # 打印这一帧的参数
  # mfcc_rep = mfcc_per.repeat(68, 1) # 将第二个维度复制68次  [68,1024*n]
  output_cat = torch.cat([lmk_per,mfcc_per], dim=-1) # [batch,T,length]
  #转换轴
  output_cat = torch.transpose(output_cat,1,0)

  # output_cat = torch.unsqueeze(output_cat, dim=0) # [1,68,3+1024*n]
  return output_cat


def extract_lankmarks(images,model):
  """
  images:[batchsize,帧数，H,W,C]  numpy
  model:提取脸部标志的模型
  """
  ## 分出每个batchsize
  lmklist = [] #[batchsize,帧数，68个点，三维坐标]
  print("开始提取关键点")  
  # 耗时
  for img_batch in images:
    # 提取图像脸部标志点
    lmklist_batch = [] #[帧数，68个点，三维坐标]
    for image in img_batch:
      lmk68 = get_landmarks(model,image)[0] 
      lmk68 = np.transpose(lmk68, (1, 0))
      lmklist_batch.append(lmk68)
      #print("1")
    lmklist.append(lmklist_batch)
  # 转为tensor，并转为与网络相同类型 
  stacked_array = np.stack(lmklist)
  #print("提取结束")
  lmklist = torch.tensor(stacked_array)
  lmklist = lmklist.type(torch.float32)
  # 去除第一帧，作为预测帧
  #print("lmk_step转化")
  lmk_step = lmklist[:,1:,:,:]
  return lmklist, lmk_step
  """output:
  lmklist:[batchsize,fps,points,dim]
  lmk_step:[batchsize,fps,points-1,dim]
  """

# input_size= [1,68,3+1024*n]
# 将input_size变成duplicated_tensor[2,batch_size,1024]
def H_reshape_input(batch_size, inputs,n):
    # 将所有第一个元素拼接在一起，并调整形状
    reshaped_input = torch.cat(first_elements, dim=0).reshape(batch_size, 68*3+1024*n)

    # dawn新加的复制堆叠到新维度上
    duplicated_tensor = torch.stack([reshaped_input, reshaped_input], dim=0)
    duplicated_tensor= duplicated_tensor.view(2,batch_size,-1)
    return reshaped_input,duplicated_tensor
