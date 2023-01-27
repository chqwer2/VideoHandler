import unittest
from SoundSplitter import *
from brian import *
from brian.hears import *
import wave
import numpy as np
import soundfile as sf

class TestSoundSplitterCases(unittest.TestCase):
    def test_hrir(self):
        self.azim = 30
        self.elev = 15

        self.my_sspl = SoundSplitter('IRCAMdb', 'tom.wav', self.azim, self.elev)
        self.my_soundsplit = self.my_sspl.soundsplit()
        [my_hrir_left, my_hrir_right] = self.my_sspl.get_hrir()

        self.data = scio.loadmat('IRC_1002_R_HRIR.mat')
        self.l_hrir = self.data['l_hrir_S']
        self.r_hrir = self.data['r_hrir_S']

        self.left = []
        self.right = []
        for i in range(0, 8192, 1):
            self.left.append(my_hrir_left[i][0])
            self.right.append(my_hrir_right[i][0])

        if self.left == self.l_hrir['content_m'][0][0][0].tolist():
            self.L = True
        else:
            self.L = False
        if self.right == self.r_hrir['content_m'][0][0][0].tolist():
            self.R = True
        else:
            self.R = False

        self.assertEqual(self.L & self.R, True)


if __name__ == '__main__':
    unittest.main()