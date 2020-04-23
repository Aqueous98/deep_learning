import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice
import numpy as np
import add-echoes as ae
import add-noise as an
import preprocessing as pp
import random
import tensorflow as tf
import scipy.signal as ss

from tensorflow.keras import layers

corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl', 'slt'])

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
    #add noise here
    #add echoes and noise here
    targets[i] = ss.stft(pp.process_sentence(sample, 16000), fs=16000, nfft=512)
    
    #randomise which sample is input
    rand = random.randint(0, 3)
    if rand==0:
        f, t, samples[i] = ss.stft(sample, fs=16000, nfft=512)
    elif rand==1:
        f, t, samples[i] = ss.stft(echosample, fs=16000, nfft=512)
    elif rand==2:
        #add just noise
        samples[i] = ss.stft(sample, fs=16000, nfft=512)
    else:
        #add both
        samples[i] = ss.stft(sample, fs=16000, nfft=512)
        
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
