
# download from https://evermeet.cx/ffmpeg
# doc https://pypi.org/project/pydub/
# pip install ffmpeg-python

from pydub import AudioSegment


# file = "/Users/haochen/Desktop/360/Shenzhen_onepark_360_VID_20230106_040915_00_004(1).mp4"

# stereo_audio = AudioSegment.from_file(file, "mp4")
# mono_audios = stereo_audio.split_to_mono()
# mono_left = mono_audios[0].export("mono_left.wav", format="wav")
# mono_right = mono_audios[1].export("mono_right.wav",format="wav")
# print("mono_audios:", mono_audios)


wavfile = "./data/audio.wav"
stereo_audio = AudioSegment.from_file(wavfile, "wav")
mono_audios = stereo_audio.split_to_mono()

mono_left = mono_audios[0].export(wavfile +"_left.wav", format="wav")
mono_right = mono_audios[1].export(wavfile +"_right.wav",format="wav")

print("Done")