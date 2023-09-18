#这是一个构成Time_frequency的图片
import numpy as np
import wave
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from brian2 import *
from sklearn import preprocessing
import numpy as np
COOKED_DIR = 'E:small/'#where are the audios
COOKED_DIR2 = 'E:/'#where you want to save your spectrograms
i = 1350
for root, dirs, files in os.walk(COOKED_DIR):
    print("Root = ", root, "dirs = ", dirs, "files = ", files)
    for filename in files:
        path_one = COOKED_DIR + filename
        f = wave.open(path_one, 'rb')
        params = f.getparams()  # 一次性返回所有的音频参数，声道数、量化位数、采样频率、采样点数
        nchannels, sampwidth, framerate, nframes = params[:4]  # 声道/量化数/采样频率/采样点数
        str_data = f.readframes(nframes)  # 指定需要读取的长度(以取样点为单位)，返回字符串类型数据
        waveData = np.frombuffer(str_data, dtype=np.int16)  # 将字符串转化为int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        plt.rcParams['figure.figsize'] = (128, 128)  #  设置figure_size尺寸6.2, 6.2
        plt.rcParams['savefig.dpi'] = 50  # 图片像素 这样输出的就是
        plt.specgram(waveData,cmap='Blues_r', NFFT=512, Fs=framerate, noverlap=500, scale_by_freq=True, sides='default')
        plt.axis('off')
        name = str(i)	# 做名字
        plt.savefig(COOKED_DIR2+"photo28"+name+".jpg", bbox_inches='tight', pad_inches=-0.1,dpi=6.25)	# 后两项为去除白边
        i += 1
        print(i)
