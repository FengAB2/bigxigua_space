import torch.nn as nn
import librosa
import torch
from configparser import ConfigParser
from synergy3DMM import SynergyNet
import numpy as np
import cv2
from FaceBoxes.FaceBoxes import main
import matplotlib.pyplot as plt

# 提取MFCC特征
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





  rotated_mesh = np.dot(R, mesh[0]) # 旋转和平移每个顶点
  rotated_mesh+= t[:,np.newaxis]
  # print('rotated_mesh.shpae',rotated_mesh.shape)

  # 将模型中心点移动到原点
  center = np.mean(rotated_mesh, axis=1)
  # print('center.shape',center.shape)
  translated_mesh = rotated_mesh - center.reshape((-1, 1))


  return translated_lmk,translated_mesh


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
  
  return mfcc

# 处理语音特征的神经网络
# audio_feature.shpae (40, 601)

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=1024, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024)
        )
# encoded_audio([1, 1024, 8])
    def forward(self, x):
        x = self.conv_layers(x)
        return x

# 处理特征点的神经网络
# translated_lmk.shpae (3, 68)
class LandmarkEncoder(nn.Module):
    def __init__(self):
        super(LandmarkEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(68*3, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 68*3),
        )

    def forward(self, x):
        translated_lmk = torch.from_numpy(np.array(x[0])).float()  # 转换为 Tensor 类型
        translated_lmk = translated_lmk.view(-1)

        x = self.encoder(translated_lmk)  # 使用 self.encoder 进行前向传播
        
        return x.view( 68, 3)  # 转换为三维数组返回


def main():
  config = ConfigParser()
  # 读取config文件
  config.read('/content/SPACE/config.ini')
  sr = config.get('mcff', 'sr')
  n_fft = config.get('mcff', 'n_fft')
  n_mfcc = config.get('mcff', 'n_mfcc')
  fps = config.get('mcff', 'fps')
  audio_feature = auido_feature_extract('/content/SPACE/data/audio/text.wav', int(sr), int(n_fft),
                                              int(n_mfcc), int(fps))
 
  audio_feature = audio_feature.reshape(-1,40,601)  # 在第一个维度上添加一个维度作为 batch size

  encoded_audios = AudioEncoder()
  encoded_audio= encoded_audios(torch.Tensor(audio_feature))
  print(encoded_audio.shape)

# 导入landmarks2
  model = SynergyNet()
  image = cv2.imread('/content/SPACE/SynergyNet/img/sample_3.jpg')
  x= get_landmarks(model,image)
  print('x[0].shpae',x[0].shape)
  translated_lmk = torch.from_numpy(np.array(x[0])).float()
  translated_lmk= translated_lmk.transpose(1,0)

  print('translated_lmk.shpae',translated_lmk.shape)
  
  # 将translated_lmk传入到neural network
  encoded_imgs= LandmarkEncoder()
  encoded_img= encoded_imgs(translated_lmk)
  print('经过neural net处理的维度',encoded_img.shape)

if __name__== '__main__':
  main()
