import wave
import struct
from numpy import array, concatenate, argmax
from numpy import abs as nabs
from scipy.signal import fftconvolve
from matplotlib.pyplot import plot, show
from math import log

def crossco(wav):
    """Returns cross correlation function of the left and right audio. It
    uses a convolution of left with the right reversed which is the
    equivalent of a cross-correlation.
    """
    cor = nabs(fftconvolve(wav[0],wav[1][::-1]))
    return cor

def trackTD(fname, width, chunksize=5000):
    track = []
    #opens the wave file using pythons built-in wave library
    wav = wave.open(fname, 'r')
    #get the info from the file, this is kind of ugly and non-PEPish
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()

    #only loop while you have enough whole chunks left in the wave
    while wav.tell() < int(nframes/nchannels)-chunksize:

        #read the audio frames as asequence of bytes
        frames = wav.readframes(int(chunksize)*nchannels)

        #construct a list out of that sequence
        out = struct.unpack_from("%dh" % (chunksize * nchannels), frames)

        # Convert 2 channels to numpy arrays
        if nchannels == 2:
            #the left channel is the 0th and even numbered elements
            left = array (list (out[0::2]))
            #the right is all the odd elements
            right = array (list  (out[1::2]))
        else:
            left = array (out)
            right = left

        #zero pad each channel with zeroes as long as the source
        left = concatenate((left,[0]*chunksize))
        right = concatenate((right,[0]*chunksize))

        chunk = (left, right)

        #if the volume is very low (800 or less), assume 0 degrees
        if abs(max(left)) < 800 :
            a = 0.0
        else:
            #otherwise computing how many frames delay there are in this chunk
            cor = argmax(crossco(chunk)) - chunksize*2
            #calculate the time
            t = cor/framerate
            #get the distance assuming v = 340m/s sina=(t*v)/width
            sina = t*340/width
            a = asin(sina) * 180/(3.14159)



        #add the last angle delay value to a list
        track.append(a)


    #plot the list
    plot(track)
    show()