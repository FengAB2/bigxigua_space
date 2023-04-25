import mcff
from configparser import ConfigParser
from get_landmark import get_outputs
config = ConfigParser()
# 读取config文件
config.read('config.ini')
sr = config.get('mcff', 'sr')
n_fft = config.get('mcff', 'n_fft')
n_mfcc = config.get('mcff', 'n_mfcc')
fps = config.get('mcff', 'fps')

audio_feature = mcff.auido_feature_extract('data/audio/text.wav', int(sr), int(n_fft),
                                           int(n_mfcc), int(fps))
print(audio_feature)

#得到3Dlandmark
lmk3d = get_outputs()
print(lmk3d)
