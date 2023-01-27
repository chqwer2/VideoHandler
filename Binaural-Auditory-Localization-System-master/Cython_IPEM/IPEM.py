# -*- coding: utf-8 -*-
import mycodecpy
from brian import *
from brian.hears import *
import numpy as np
import wave
import os

# Input arguments:
#   inInputFileName = name of the file to process (only .wav files)
#   inInputFilePath = directory path to the input file
#                     if empty or not specified, '' is used by default
#   inOutputFileName = name for the output file
#                      if empty or not specified, 'nerve_image.ani' is used by default
#   inOutputFilePath = directory path to the output file
#                      if empty or not specified, '' is used by default
#   inSampleFrequency = sample frequency of the wave file (in Hz)
#                       if empty or not specified, 22050 is used by default
#   inNumOfChannels = number of auditory channels
#                     if empty or not specified, 40 is used by default
#   inFirstFreq = center frequency for the first auditory channel (in cbu)
#                 if empty or not specified, 2.0 is used by default
#   inFreqDist = distance between center frequencies of two adjecent auditory channels (in cbu)
#                if empty or not specified, 0.5 is used by default

class IPEM:
    # 設定初值
    def __init__(self, inInputFileName, inInputFilePath, inOutputFileName, inOutputFilePath):
        self.inNumOfChannels = 40
        self.inFirstFreq = 2.0
        self.inFreqDist = 0.5
        self.inInputFileName = inInputFileName
        self.inInputFilePath = inInputFilePath
        self.inOutputFileName = inOutputFileName
        self.inOutputFilePath = inOutputFilePath
        self.inSoundFileFormat = 2
        self.inputfile = self.inInputFilePath+self.inInputFileName
        self.outputfile = self.inOutputFilePath+self.inOutputFileName

    # 檢查輸入檔案
    def checkInputFile(self):
        if os.path.splitext(self.inputfile)[-1] != '.wav':
            print 'Only .wav files'
            sys.exit(0)

        self.wave = wave.open(self.inputfile)
        self.inSampleFrequency = self.wave.getframerate()

    # IPEM
    def ipem(self):
        self.checkInputFile()
        result = mycodecpy.callCfunc(self.inNumOfChannels, self.inFirstFreq, self.inFreqDist,
                             self.inInputFileName, self.inInputFilePath, self.inOutputFileName,
                             self.inOutputFilePath, self.inSampleFrequency, self.inSoundFileFormat)

    # 繪圖
    def plot(self):
        sound = loadsound(self.inputfile)

        result = []
        with open(self.outputfile,'r') as f:
            for line in f:
                result.append(line.split(' ')[:-1])
        result = np.transpose(result)
        result = result.astype(np.float)

        [Rows,Cols] = result.shape
        inMinY = np.min(result)
        inMaxY = np.max(result)
        Scale = abs(inMaxY-inMinY)

        figure(1)
        for i in range(Rows):
            if (Scale != 0):
                title('Auditory nerve image')
                plot((i+0.05) + 0.9*(result[i][:]-inMinY), color="blue")
            else:
                title('Auditory nerve image')
                plot((i+0.05) + 0.9*(result[i][:]-inMinY), color="blue")
        # xlabel('Time', fontsize=18);ylabel('Freq. Channels', fontsize=18)
				
        figure(2)
        title('Audio Signal')
        plot(sound.times, sound)
        xlabel('Time (s)')
        ylabel('Amplitude')
        show()

# 執行
if __name__ == '__main__':
    word='coffee';angle='0';channel='L'
    inInputFileName = word+'_'+str(angle)+channel+'.wav'
    inInputFilePath = 'wav/'
    inOutputFileName = word+'_'+str(angle)+channel+'.txt'
    inOutputFilePath = 'txt/'
    myIPEM = IPEM(inInputFileName, inInputFilePath, inOutputFileName, inOutputFilePath)
    myIPEM.ipem()
    # myIPEM.plot()
