# -*- coding: utf-8 -*-
from brian import *
from brian.hears import *
import soundfile as sf
# import numpy as np

class SoundSplitter:
    # 設定初值
    def __init__(self, path, filename, azim, elev):
        self.path = path
        self.file = filename
        self.azim = azim
        self.elev = elev

    # 取得 HRIR
    def hrir(self):
        self.hrtfdb = IRCAM_LISTEN(self.path)
        self.hrtfset = self.hrtfdb.load_subject(1050)

        if self.azim>180:
            self.hrir = self.hrtfset(azim = np.abs(self.azim-360), elev = self.elev)
            self.left_hrir = self.hrir.right
            self.right_hrir = self.hrir.left
        else:
            self.hrir = self.hrtfset(azim = self.azim, elev = self.elev)
            self.left_hrir = self.hrir.left
            self.right_hrir = self.hrir.right

    # 輸入原始聲音
    def sound(self):
        self.sound = loadsound(self.file)

        # Only Mono
        if self.sound.nchannels != 1:
            print 'Just mono files'
            sys.exit(0)

    # 分為左右耳聲道
    def soundsplit(self):
        self.hrir()
        self.sound()

        self.set_hrir = HRTF(self.left_hrir, self.right_hrir)
        self.output = self.set_hrir.apply(self.sound)

        self.ofL = self.file.replace(".wav", "_"+str(self.azim)+"L.wav")
        self.ofR = self.file.replace(".wav", "_"+str(self.azim)+"R.wav")
        self.output.left.save(self.ofL)
        self.output.right.save(self.ofR)

    # 繪圖
    def plot(self):
        figure(1)
        subplot(2, 1, 1)
        plot(self.left_hrir.times, self.left_hrir)
        title('Left HRIR')
        subplot(2, 1, 2)
        plot(self.right_hrir.times, self.right_hrir)
        title('Right HRIR')
		# plt.xlabel('time', fontsize=18)

        figure(2)
        plot(self.sound.times, self.sound)
        title('Original sound wave')
		# plt.xlabel('time', fontsize=18)

        figure(3)
        subplot(2, 1, 1)
        plot(self.output.left.times, self.output.left)
        title('Left channel')
        subplot(2, 1, 2)
        plot(self.output.right.times, self.output.right)
        title('Right channel')
		# plt.xlabel('time', fontsize=18)
        show()

        # play(self.sound)
        # play(self.output)

    # 回傳
    def get_hrir(self):
        return self.left_hrir , self.right_hrir

    def get_output(self):
        return self.output.left, self.output.right


# 執行
if __name__ == '__main__':
    sspl = SoundSplitter('IRCAMdb', 'violin.wav', 0, 0)
    sspl.soundsplit()
    # sspl.plot()

    # print sspl.left_hrir
    # print sspl.right_hrir
    # print sspl.sound
    # print sspl.output

