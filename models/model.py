import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
from configparser import ConfigParser


  
# 音频编码器
# 经过神经网络处理后x.shape torch.Size([601, 1024, 1])
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        # 设置输入的特征维度
        in_channels = 1

        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 1024, kernel_size=3, stride=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
        )

        # 使用 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        # 前向传播
        x = self.conv_layers(x)
        
        return x

        # input: (batch_size, 1, audio_length)

        # output: (batch_size, 1024, encoded_length)





# 网络层

# 定义一个基本的全连接块
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True, use_relu=True, init_method="xavier"):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.relu = nn.LeakyReLU(inplace=True) if use_relu else None
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init_method == "he":
            nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 定义68个3DDFA关键点的编码器
class Encoder68(nn.Module):
    def __init__(self, init_method="xavier"):
        super(Encoder68, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(68*3, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 68*3, use_bn=False, use_relu=False, init_method=init_method)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.layers(x)
        return  x  #输出是(batch,68*3)

# 定义52个MTCNN眼睛关键点的编码器
class Encoder52(nn.Module):
    def __init__(self, init_method="xavier"):
        super(Encoder52, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(104, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 1024, init_method=init_method),
            LinearBlock(1024, 52*2, use_bn=False, use_relu=False, init_method=init_method)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class S2L_LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(S2L_LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, h, c):
        '输入数据维度为 (sequence_length, batch_size, input_size)'
        out, (hidden, cell) = self.lstm(x, (h, c))
        out = self.fc(out)
        return out, hidden, cell

# 回归器网络

# 输入的是duplicated_tensor[batch_size,3*68+1024*n]
class REGRESSOR(nn.Module):
    def __init__(self,input_size,hidden_size):
      super().__init__()
      self.fc1= nn.Linear(input_size,hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, hidden_size)
      self.fc4 = nn.Linear(hidden_size, hidden_size)
      def forward(self,x):
        x= x.view(-1,x.size(-1))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 2, self.fc4.out_features)
        return x
# def initialize_hidden_state(duplicated_tensor, batch_size, n):
#     input_size = 3*68 + 1024*n
#     hidden_size = 1024
#     num_layers = 2

#     # 定义初始化器
#     initializer = REGRESSOR(input_size, hidden_size)

#     # 获得初始时间步的输入
#     initial_inputs = duplicated_tensor[:, 0]

#     # 使用初始化器获得初始隐藏状态
#     hidden = initializer(initial_inputs)
#     return hidden

