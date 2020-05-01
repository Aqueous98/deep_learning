import soundfile as sf
import numpy as np

def add_noise(data_new, samplerate, data_noise, samplerate_noise):

	j=0
	for i in range(len(data_new)):
		data_new[i] = data_new[i] + data_noise[j,0]*6  #6 is noise strength
		if samplerate < samplerate_noise:
			j=j+2
		else:
			j=j+1

	

	#sf.write('new_file.flac', data_new, samplerate)

	return(data)

def add_noise_gaussian(sample):
	noise = np.random.normal(0, 1, sample.size)
	return sample+noise
	
#path = "84-121123-0000.flac"
#path = "arctic_a0001.wav"
#path_noise = "rain_noise.flac"

#data, samplerate = sf.read(path) ##Main File
#data_noise, samplerate_noise = sf.read(path_noise)
#print(samplerate,samplerate_noise)

#add_noise(data,samplerate, data_noise,samplerate_noise)
