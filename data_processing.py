import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

import librosa

from copy import deepcopy

# Globals
corpus = pp.download_corpus(download_flag=False, speaker=['bdl', 'slt'])
max_val = 0
n_fft = 512


def pad(data, len):
  return librosa.util.fix_length(data, len)


def loss(target, pred):
  return tf.keras.losses.MSE(target, pred)


# dimensionality of input = number of frequency channels (257)
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(100, input_shape=(max_val, n_fft//2 + 1)))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.Dense(257))

model.compile(optimizer='adam', loss=loss)

samples = []
targets = []

for i in range(len(corpus)):
  if len(corpus[i].data) > max_val:
    max_val = len(corpus[i].data)

for i in range(len(corpus)):
  data = corpus[i].data.astype(np.float64)
  sample = deepcopy(data)
  fs = corpus[i].fs
  echosample = pad(ae.add_echoes(sample), max_val)
  noisesample = pad(an.add_noise(sample, sf.read("RainNoise.flac")), max_val)
  bothsample = pad(ae.add_echoes(noisesample), max_val)

  # data_out, _ = pp.process_sentence(sample, fs=fs)
  # data_out = deepcopy(sample)
  data = pad(data, max_val)
  target_fft = np.abs(librosa.core.stft(data, n_fft=n_fft).T)
  targets.append(target_fft)  # np.abs(ss.stft(data_out, fs=fs, nfft=n_fft))

  #randomise which sample is input
  rand = random.randint(0, 2)
  if rand == 0:
    samples.append(
      np.abs(librosa.core.stft(echosample,
                               n_fft=n_fft,
                               center=True).T)
    )
  elif rand == 1:
    samples.append(
      np.abs(librosa.core.stft(noisesample,
                               n_fft=n_fft,
                               center=True).T)
    )
  else:
    samples.append(
      np.abs(librosa.core.stft(bothsample,
                               n_fft=n_fft,
                               center=True).T)
    )

samples = np.array(samples)
targets = np.array(targets)

print(samples.shape)
print(targets.shape)
#train the network
model.fit(samples, targets, epochs=1, batch_size=2)

#test
input = pad(corpus[0].data.astype(np.float64), max_val)
# sounddevice.play(input, 16000)
# plt.figure()
# plt.plot(input)
# plt.show()

test = np.abs(librosa.core.stft(input, n_fft=n_fft, center=True).T)
test = test.reshape(1, test.shape[0], test.shape[1])
print(test.shape)
#run the network on the test sample
output = model.predict(test)
print(output.shape)
output_lowdim = output[0]
print(output_lowdim.shape)
output_sound = librosa.griffinlim(output_lowdim.T)
#plot and play output
sounddevice.play(output_sound, 16000)
plt.plot(output_sound)
plt.show()
