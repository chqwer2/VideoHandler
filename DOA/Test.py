import wave
import struct
from numpy import array, concatenate, argmax
from numpy import abs as nabs
from scipy.signal import fftconvolve
from matplotlib.pyplot import plot, show
from math import log
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss

def crossco(wav):
    """Returns cross correlation function of the left and right audio. It
    uses a convolution of left with the right reversed which is the
    equivalent of a cross-correlation.
    """
    # wav[1][::-1]
    cor = nabs(fftconvolve(wav[0], wav[1][::-1]))
    return cor

def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array,Angles[i])
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
    return DoAsMUSIC,pspectrum

def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = LA.eig(CovMat)
    S = U[:,0:L]
    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
    return DoAsESPRIT


def trackTD(fname, width, chunksize=50):
    track = []
    #opens the wave file using pythons built-in wave library
    wav = AudioSegment.from_file(fname, "wav")
    mono_audios = wav.split_to_mono()

    mono_left = np.array(mono_audios[0].get_array_of_samples())[:10000].astype(np.float32)
    mono_right = np.array(mono_audios[1].get_array_of_samples())[:10000].astype(np.float32)

    mono_left = np.array(AudioSegment.from_file("../data/mono_left.wav").get_array_of_samples())[:10000].astype(np.float32)
    mono_right = np.array(AudioSegment.from_file("../data/mono_right.wav").get_array_of_samples())[:10000].astype(np.float32)


    nchannels, sampwidth, framerate, nframes = wav.channels, wav.sample_width, wav.frame_rate, wav.frame_count()

    # 2 microphones would only tell you whether a sound is coming from the right or left. The reason your 2 ears can figure out whether a sound is coming from in front of, or behind you, is because the outer structure of your ear modifies the sound depending on the direction, which your brain interprets and then corrects for.

    # .frame_width
    #get the info from the file, this is kind of ugly and non-PEPish
    # (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()

    #only loop while you have enough whole chunks left in the wave

    for i in tqdm(range(int(nframes/nchannels)-chunksize)):

        #read the audio frames as asequence of bytes
        left = mono_left[i: i+chunksize]
        right = mono_right[i: i+chunksize]

        # print("Read frame:", left, right)

        #zero pad each channel with zeroes as long as the source
        # left = concatenate((left, [0]*chunksize))
        # right = concatenate((right, [0]*chunksize))
        X_STEP = 100
        x = np.linspace(-0.8, 0.8, X_STEP)
        nmicro = 2
        A_STEP = 100
        p = np.zeros(x.shape[0])  # 声强谱矩阵
        pos = np.array([0, -0.2]) #这里使用的是电脑麦克风,2个麦克风间距约为2分米
        chunk = np.array([left, right])
        # print("chunk :", chunk.shape)
        #
        data_n = np.fft.fft(chunk) / chunk.shape[1]
        # data_n = data_n[:, :data.shape[1] // 2]  # 取前一半频率
        #
        # data_n[:, 1:] *= 2  # 将频率范围内的频率幅值翻倍

        r = np.zeros((A_STEP, nmicro, nmicro), dtype=complex)

        for fi in range(1, A_STEP + 1):
            # 计算每个频率下的R矩阵
            # 自相关函数
            rr = np.dot(data_n[:, fi * 10 - 10:fi * 10 + 10],
                        data_n[:, fi * 10 - 10:fi * 10 + 10].T.conjugate()) / nmicro
            r[fi - 1, ...] = np.linalg.inv(rr)

        # MVDR搜索过程
        for i in range(x.shape[0]):
            dm = np.sqrt(x[i] ** 2 + 1)
            delta_dn = pos * x[i] / dm
            # 遍历角度
            for fi in range(1, A_STEP + 1):
                # 计算导向向量
                a = np.exp(-1j * 2 * np.pi * fi * 100 * delta_dn / 340)
                p[i] = p[i] + 1 / np.abs(np.dot(np.dot(a.conjugate(), r[fi - 1]), a))  # 计算每个频率下的声强谱

        p /= np.max(p)
        # 获取前5个最大值
        a = np.argsort(p)[-5:]
        print(a)

        # 输出声源所在的方向
        # print(np.average(p_max))


        #if the volume is very low (800 or less), assume 0 degrees
        if abs(max(left)) < 800:
            a = 0.0
        else:

            L = 2  # number of sources fix resource
            N = 32  # number of ULA elements


            # DoAsESPRIT = esprit(CovMat, L, N)


            #otherwise computing how many frames delay there are in this chunk
            # cor = argmax(crossco(chunk)) #- chunksize*2
            # #calculate the time
            # t = cor/framerate
            # sina = t*340/width
            # a = np.arcsin(sina) * 180/(3.14159)
            # print(a)


        #add the last angle delay value to a list
        track.append(a)


    #plot the list
    plot(track)
    show()


if __name__ == "__main__":
    fname = "../data/pano.wav"
    # fname = "../data/norma;.wav"
    width = 0.02
    trackTD(fname, width, chunksize=5000)