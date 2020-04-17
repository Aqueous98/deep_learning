import os, argparse, sys
from pathlib import Path

import numpy as np

from scipy.interpolate import interp1d

import scipy.io as sio
from scipy.signal import spectrogram, periodogram, welch
from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.datasets import Sample, Dataset, CMUArcticCorpus
from pyroomacoustics.datasets import CMUArcticSentence

import librosa
from librosa.core import power_to_db, db_to_power, amplitude_to_db, stft, istft, fft_frequencies, db_to_amplitude, power_to_db, db_to_power
from librosa.display import specshow

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory


def get_audiogram(x, y, order='cubic'):
  """
  'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 
  'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline 
  interpolation of zeroth, first, second or third order; 'previous' and 'next' 
  simply return the previous or next value of the point) or as an integer 
  specifying the order of the spline interpolator to use. Default is 'linear'
  """
  X = sorted(x)
  Y = [i for _, i in sorted(zip(x, y))]

  if float(X[0]) is not 0.:
    tempX = X
    tempY = Y
    X = [0.]
    Y = [tempY[0]]
    X.extend(tempX)
    Y.extend(tempY)

  if float(X[-1]) is not 8000.:
    X.extend([8000.])
    Y.extend([Y[-1]])

  z = interp1d(X, Y, kind=order)

  return z, (X, Y)


def process_audiogram(x_audiogram, y_audiogram, freq):

  audiogram_eq, (x_audiogram, y_audiogram) = get_audiogram(x_audiogram, y_audiogram, 'linear')

  x_range_audiogram = freq[freq >= np.min(x_audiogram)]
  x_range_audiogram = x_range_audiogram[
    x_range_audiogram <= np.max(x_audiogram)]

  audiogram = np.array([audiogram_eq(x) for x in x_range_audiogram])
  audiogram = (x_range_audiogram, audiogram)

  plot_audiogram(audiogram)

  return audiogram


def plot_audiogram(audiogram, modulated=None):
  f = plt.figure(1)
  plt.title('Audiogram')
  plt.grid(True)

  # plt.plot(source[0], source[1], 'r-')
  plt.plot(audiogram[0], audiogram[1], 'b-')
  # if modulated is not None:
  # plt.plot(modulated[0], modulated[1], 'g-')

  plt.legend(
    ['audiogram']  # ['source, audiogram, modulated']
    # if modulated is not None else ['audiogram']
  )

  plt.ylim((140, -20))
  plt.xlim((0, 8 * 1e3))
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Amplitude [dB]')
  # plt.show()


def process_sentence(data, fs):

  # Default settings for speech analysis
  # n_fft = 512 to provide 25ms-35ms samples
  # (https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)
  n = len(data)
  n_fft = 512
  center = True

  # Pad the data since istft will drop any data in the last frame if samples are
  # less than n_fft.
  data_pad = librosa.util.fix_length(data, n + n_fft//2)

  # Get the frequency distribution
  freq = fft_frequencies(sr=fs, n_fft=n_fft)

  # Get the equation and freq, db array from the audiogram provided
  x_audiogram = [125, 250, 500, 1000, 1500, 2000, 4000]
  y_audiogram = [40, 35, 40, 65, 85, 105, 110]
  audiogram = process_audiogram(x_audiogram, y_audiogram, freq)

  # Perform the stft, separate magnitude and save phase for later (important)
  data_pad_stft = stft(data_pad, n_fft=n_fft, center=center)
  mag, phase = librosa.core.magphase(data_pad_stft)

  # Get the minimum and maximum scaling factor and save for later (important)
  min_scale, max_scale = np.min(mag), np.max(mag)
  scale = max_scale - min_scale

  # Normalize the magnitude
  mag = librosa.util.normalize(mag)
  s_db = amplitude_to_db(mag)

  # Scale sample to 120 db to map to the audiogram
  s_db += 120

  for i, row in enumerate(s_db):
    for idx, (sig, thresh) in enumerate(zip(row, audiogram[1])):
      if np.min(audiogram[1]) >= freq[idx] <= np.max(audiogram[1]):
        s_db[i][idx] = thresh

  # Reduce magnitude
  s_db -= 120

  f = plt.figure(2)
  specshow(s_db, x_axis='time', y_axis='linear')
  plt.colorbar()

  # Convert db back to amplitude
  data_mod_pow = db_to_amplitude(s_db)

  # Apply scaling to obtain initial signal
  # data_mod_pow = data_mod_pow*scale + min_scale

  # Multiply new magnitude with saved phase to reconstruct sentence
  data_orig_mod = data_mod_pow * phase

  # Perform the inverse stft
  data_mod = istft(data_orig_mod, center=center, length=n)

  return data_mod


if __name__ == '__main__':

  # Download the corpus, be patient
  download_flag = True
  if os.path.exists(ARCTIC_DIR):
    download_flag = False

  corpus = CMUArcticCorpus(
    basedir=ARCTIC_DIR,
    download=download_flag,
    speaker=['clb']
  )

  # Pick a sentence
  sentence_idx = 3

  # Get the timeseries and sampling frequency
  data = corpus[sentence_idx].data.astype(float)
  fs = corpus[sentence_idx].fs

  data_mod = process_sentence(data, fs)

  corpus[sentence_idx].data = data_mod
  corpus[sentence_idx].play()
  plt.show()
