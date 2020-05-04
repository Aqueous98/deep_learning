import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice
import numpy as np
import add_echoes as ae
import add_noise as an
import preprocessing as pp
import random
import tensorflow as tf
import scipy.signal as ss
import soundfile as sf

from tensorflow.keras import layers

corpus = pp.download_corpus(download_flag=False, speaker=['bdl', 'slt'])

print(len(corpus))

# dimensionality of input = number of frequency channels (257)
model = tf.keras.Sequential()
model.add(layers.Dense(100, input_shape=(257,)))
model.add(layers.LSTM(100))
model.add(layers.Dense(257))

def loss(target, pred):
  return tf.keras.losses.MSE(target, pred)
model.compile(optimizer='adam', loss=loss)

samples = corpus
targets = corpus

for i in len(corpus):
    sample = corpus[i].data
    echosample = ae.add_echoes(sample)
    noisesample = an.add_noise(sample, sf.read("RainNoise.flac"))
    bothsample = ae.add_echoes(noisesample)
    targets[i] = ss.stft(pp.process_sentence(sample, 16000), fs=16000, nfft=512)
    
    #randomise which sample is input
    rand = random.randint(0, 2)
    if rand==0:
        f, t, samples[i] = ss.stft(echosample, fs=16000, nfft=512)
    elif rand==1:
        f, t, samples[i] = ss.stft(noisesample, fs=16000, nfft=512)
    else:
        samples[i] = ss.stft(bothsample, fs=16000, nfft=512)
        
#train the network
model.fit(samples, targets, epochs = 50)

#test
input = corpus[len(corpus)-1]
sounddevice.play(input, 16000)
plt.figure()
plt.plot(input)
plt.show()

#run the network on the test sample
output = model.predict(ss.stft(input, fs=16000, nfft=512))
#plot and play output
sounddevice.play(ss.istft(output, fs=16000, nfft=512), 16000)
plt.plot(output)
plt.show()
