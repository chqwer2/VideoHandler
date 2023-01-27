import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import sys

import matplotlib.pyplot as plt
n_mels = 64
n_frames = 5
n_fft = 1024
hop_length = 512
power = 2.0

hop_length = 1024
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

# 读取音频wav文件
audio_path = r"pano.wav_left.wav"
y, sr = librosa.load(audio_path, sr=None, mono=True)
D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
                            ref=np.max)
img = librosa.display.specshow(D1, y_axis='log', sr=sr, hop_length=hop_length,
                         x_axis='time', ax=ax[1])
ax[1].set(title='left power spectrogram')
ax[1].label_outer()

# 读取音频wav文件
audio_path = r"pano.wav_right.wav"
y, sr = librosa.load(audio_path, sr=None, mono=True)
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
                            ref=np.max)
img = librosa.display.specshow(D2, y_axis='log', sr=sr, hop_length=hop_length,
                         x_axis='time', ax=ax[0])
ax[0].set(title='right power spectrogram')
ax[0].label_outer()
print("D1:", D1)

# 读取音频wav文件
img = librosa.display.specshow(D1-D2, y_axis='mel', sr=sr, hop_length=hop_length,
                         x_axis='time', ax=ax[2])
ax[2].set(title='power spectrogram difference')
ax[2].label_outer()

fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()
