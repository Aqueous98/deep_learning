import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import random

import os, argparse, sys, shutil
from pathlib import Path

import numpy as np

from scipy.interpolate import interp1d

import scipy.io as sio
import scipy.signal
from scipy.signal import spectrogram, periodogram, welch, wiener
from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.datasets import Sample, Dataset, CMUArcticCorpus
from pyroomacoustics.datasets import CMUArcticSentence

import librosa
from librosa.core import power_to_db, db_to_power, amplitude_to_db, stft, istft
from librosa.core import fft_frequencies, db_to_amplitude, power_to_db
from librosa.core import db_to_power
from librosa.display import specshow
import librosa.filters
import librosa.effects

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directory, PP_DATA_DIR
from constants import NOISE_DIR

import add_echoes as ae
import add_noise as an
import soundfile as sf

from copy import deepcopy

from preprocessing import process_sentence, download_corpus

root_path = 'data/'

# Global vars to change here
NFFT = 512
BATCH_SIZE = 32
EPOCHS = 50
max_val = 1

# Model Parameters
METRICS = ['mean_absolute_error', 'accuracy']
LOSS = 'mean_absolute_error'
OPTIMIZER = 'adam'


def pad(data, length):
  return librosa.util.fix_length(data, length)


def gen_model(input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)):
  """ Define the model architecture """

  model = Sequential()
  model.summary()

  return model


def model_load(model_name='speech2speech'):
  """ Load a saved model if present """
  json_file = open(root_path + 'models/{}/model.json'.format(model_name), 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  # Load model weights
  model = model_from_json(loaded_model_json)
  model.load_weights(root_path + 'models/{}/model.h5'.format(model_name))

  # Compile the model
  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

  return model


def generate_dataset(truth_type=None):
  """ 
  Save input and processed files for tests
  
  @param truth_type None for both, 'raw' for non processed and 'eq' for equalized
  """

  corpus = download_corpus()
  if not os.path.exists(PP_DATA_DIR):
    create_preprocessed_dataset_directory()
  else:
    corpus_len = len(corpus)
    max_val = 0
    # Find maximum length of time series data to pad
    for i in range(corpus_len):
      if len(corpus[i].data) > max_val:
        max_val = len(corpus[i].data)

    is_eq = False
    is_raw = False
    is_both = True

    if truth_type == 'eq':
      is_eq = True
      is_both = False
    elif truth_type == 'raw':
      is_raw = True
      is_both = False

    X = []
    y_eq = None
    y_raw = None

    if is_eq or is_both:
      y_eq = []
    if is_raw or is_both:
      y_raw = []

    # Get each sentence from corpus and add random noise/echos/both to the input
    # and preprocess the output. Also pad the signals to the max_val
    for i in range(corpus_len):
      # Original data in time domain
      data_orig_td = corpus[i].data.astype(np.float64)
      yi = pad(deepcopy(data_orig_td), max_val)
      # Sampling frequency
      fs = corpus[i].fs

      # Pad transformed signals
      echosample = ae.add_echoes(data_orig_td)
      noisesample = an.add_noise(
        data_orig_td,
        sf.read(os.path.join(NOISE_DIR,
                             "RainNoise.flac"))
      )
      bothsample = ae.add_echoes(noisesample)

      echosample = pad(echosample, max_val)
      noisesample = pad(noisesample, max_val)
      bothsample = pad(bothsample, max_val)

      # Equalize data for high frequency hearing loss
      data_eq = None
      if is_eq or is_both:
        data_eq, _ = process_sentence(yi, fs=fs)
        yi_stft_eq = librosa.core.stft(data_eq, n_fft=NFFT)
        y_eq.append(np.abs(yi_stft_eq.T))

      # Use non processed input and pad as well
      data_raw = None
      if is_raw or is_both:
        data_raw = deepcopy(yi)
        yi_stft_raw = librosa.core.stft(data_raw, n_fft=NFFT)
        y_raw.append(np.abs(yi_stft_raw.T))

      #randomise which sample is input
      rand = random.randint(0, 2)
      if rand == 0:
        echosample_stft = librosa.core.stft(
          echosample,
          n_fft=NFFT,
          center=True
        )
        X.append(np.abs(echosample_stft.T))
      elif rand == 1:
        noisesample_stft = librosa.core.stft(
          noisesample,
          n_fft=NFFT,
          center=True
        )
        X.append(np.abs(noisesample_stft.T))
      else:
        bothsample_stft = librosa.core.stft(
          bothsample,
          n_fft=NFFT,
          center=True
        )
        X.append(np.abs(bothsample_stft.T))

    # Convert to np arrays
    if is_eq or is_both:
      y_eq = np.array(y_eq)

    if is_raw or is_both:
      y_raw = np.array(y_raw)

    X = np.array(X)

    # Save files
    np.save(os.path.join(PP_DATA_DIR, "inputs.npy"), X, allow_pickle=True)

    if is_eq or is_both:
      np.save(
        os.path.join(PP_DATA_DIR,
                     "truths_eq.npy"),
        y_eq,
        allow_pickle=True
      )
    if is_eq or is_both:
      np.save(
        os.path.join(PP_DATA_DIR,
                     "truths_raw.npy"),
        y_raw,
        allow_pickle=True
      )

    if is_eq and not is_both:
      return X, y_eq, None
    elif is_raw and not is_both:
      return X, y_raw, None

  return X, y_raw, y_eq


def load_dataset(truth_type='raw'):
  """ 
  Load a dataset and return the normalized test and train datasets
   appropriately split test and train sets.
  
  @param truth_type 'raw' for non processed and 'eq' for equalized
  """

  X = None
  y = None

  X_train = None
  X_test = None
  X_val = None

  y_test = None
  y_train = None
  y_val = None

  if not os.path.isfile(PP_DATA_DIR + 'inputs.npy'):
    X, y, _ = generate_dataset(truth_type)
  else:
    X = np.load(os.path.join(PP_DATA_DIR, 'inputs.npy'))
    y = np.load(os.path.join(PP_DATA_DIR, 'truths.npy'))

  # Generate training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_model(model, model_name='speech2speech'):

  model_json = model.to_json()
  if not os.path.exists(root_path + 'models/{}'.format(model_name)):
    os.makedirs(root_path + 'models/{}'.format(model_name))
  with open(
    root_path + 'models/{}/model.json'.format(model_name),
    'w'
  ) as json_file:
    json_file.write(model_json)

  model.save_weights(root_path + 'models/{}/model.h5'.format(model_name))

  return model


def test_and_train(model_name='speech2speech', retrain=True):
  """ 
    Test and/or train on given dataset 

    @param model_name name of model to save.
    @param retrain True if retrain, False if load from pretrained model
  """

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(truth_type='raw')
  X = X_train
  y = y_train
  model = None

  # Scale the sample X and get the scaler
  # scaler = scale_sample(X)

  # Check if model already exists and retrain is not being called again
  if (
    os.path.isfile(root_path + 'models/{}/model.json'.format(model_name))
    and not retrain
  ):
    model = model_load()
  else:
    X = X_train
    y = y_train

    print("x_train shape:", X_train.shape, "y_train shape:", y_train.shape)

    # X_train_norm = normalize_sample(X_train).reshape(
    #   [-1,
    #    X_train.shape[1],
    #    X_train.shape[2],
    #    1]
    # )
    # X_val_norm = normalize_sample(X_val).reshape(
    #   [-1,
    #    X_val.shape[1],
    #    X_val.shape[2],
    #    1]
    # )
    model = None

    model = gen_model(
      (X_train.shape[1],
       X_train.shape[2],
       1),
      y_train.shape[1]
    )
    print('Created Model...')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print('Compiled Model...')

    # fit the keras model on the dataset
    cbs = [
      callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=100,
        verbose=1,
        mode='auto'
      )
    ]

    # model.fit(
    #   X_train_norm,
    #   y_train,
    #   epochs=EPOCHS,
    #   batch_size=BATCH_SIZE,
    #   validation_data=(X_val_norm,
    #                    y_val),
    #   verbose=1,
    #   callbacks=cbs
    # )
    print('Model Fit...')

    model = save_model(model, model_name)

  X = X_test
  y = y_test

  # X = normalize_sample(X_test).reshape(
  #   [-1,
  #    X_test.shape[1],
  #    X_test.shape[2],
  #    1]
  # )

  model = model_load(model_name)

  _, mse, accuracy = model.evaluate(X, y, verbose=0)
  print('Accuracy: %.2f' % (accuracy*100), mse)

  # y_pred = predict_values(scaler.fit_transform(X), model)

  return


if __name__ == '__main__':
  # test_and_train(model_name='speech2speech', retrain=True)
  generate_dataset()
