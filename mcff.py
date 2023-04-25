import librosa
from configparser import ConfigParser


# 提取MFCC特征


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
  # 将mfcc转化成(帧数,1,40)添加一个通道维度
  mfcc= mfcc.reshape(-1,1,40)
  return mfcc

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
  
  print('audio_feature_new.shpae',audio_feature.shape)
if __name__=='__main__':
  main()