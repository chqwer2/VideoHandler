cdef extern from "audiprog.h":
	cdef long AudiProg (long, double, double,
			const char*, const char*,
			const char*, const char*,
			double, long)
	
def callCfunc (long inNumOfChannels, double inFirstFreq, double inFreqDist,
			const char* inInputFileName, const char* inInputFilePath,
			const char* inOutputFileName, const char* inOutputFilePath,
			double inSampleFrequency, long inSoundFileFormat):
	AudiProg (inNumOfChannels, inFirstFreq, inFreqDist,
			inInputFileName, inInputFilePath,
			inOutputFileName, inOutputFilePath,
			inSampleFrequency, inSoundFileFormat)

