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

corpus = pp.download_corpus(download_flag=False, speaker=['bdl', 'slt'])

print(len(corpus))

# dimensionality of input = number of frequency channels (257)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, input_shape=(257,)))
# model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(257))


def loss(target, pred):
  return tf.keras.losses.MSE(target, pred)


model.compile(optimizer='adam', loss=loss)

samples = []
targets = []

max_val = 0
for i in range(len(corpus)):
  if len(corpus[i].data) > max_val:
    max_val = len(corpus[i].data)

for i in range(len(corpus)):
  sample = corpus[i].data.astype(np.float64)
  fs = corpus[i].fs
  echosample = librosa.util.fix_length(ae.add_echoes(sample), max_val)
  noisesample = librosa.util.fix_length(
    an.add_noise(sample,
                 sf.read("RainNoise.flac")),
    max_val
  )
  bothsample = librosa.util.fix_length(ae.add_echoes(noisesample), max_val)

  # data_out, _ = pp.process_sentence(sample, fs=fs)
  # data_out = deepcopy(sample)
  target_fft = np.abs(librosa.core.stft(sample, n_fft=512))
  targets.append(target_fft)  # np.abs(ss.stft(data_out, fs=fs, nfft=512))

  #randomise which sample is input
  rand = random.randint(0, 2)
  if rand == 0:
    samples.append(
      np.abs(librosa.core.stft(echosample,
                               n_fft=512,
                               center=True))
    )
  elif rand == 1:
    samples.append(
      np.abs(librosa.core.stft(noisesample,
                               n_fft=512,
                               center=True))
    )
  else:
    samples.append(
      np.abs(librosa.core.stft(bothsample,
                               n_fft=512,
                               center=True))
    )

samples = np.array(samples)
targets = np.array(targets)

#train the network
model.fit(samples, targets, epochs=50)

#test
input = corpus[len(corpus) - 1]
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
