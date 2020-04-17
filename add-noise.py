import soundfile as sf
import numpy as np

def add_noise(data_new, samplerate, data_noise, samplerate_noise):

	j=0
	for i in range(len(data_new)):
		data_new[i] = data_new[i] + data_noise[j,0]*6
		if samplerate < samplerate_noise:
			j=j+2
		else:
			j=j+1

	

	#sf.write('new_file.flac', data_new, samplerate)

	return(data)
	
