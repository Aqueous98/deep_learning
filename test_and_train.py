import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import random

import timeit

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
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs

import add_echoes as ae
import add_noise as an
import soundfile as sf

import sounddevice

from copy import deepcopy

from preprocessing import process_sentence, download_corpus

root_path = 'data/'

# Global vars to change here
NFFT = 512
FS = 16000
BATCH_SIZE = 32
EPOCHS = 10
max_val = 1

CHUNK = 500

# Model Parameters
METRICS = ['mse', 'accuracy']
LOSS = 'mse'
OPTIMIZER = 'adam'


def pad(data, length):
  return librosa.util.fix_length(data, length)


def normalize_sample(X):
  """ Normalize the sample """

  X = X / np.linalg.norm(X)

  return X


def gen_model(input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)):
  """ Define the model architecture """
  output_shape = input_shape[2]
  model = Sequential()

  model.add(LSTM(100, return_sequences=True))
  model.add(Dense(output_shape))

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


def save_distributed_files(truth_type=None, corpus=None):
  """ 
  Save input and processed files for tests
  
  @param truth_type None for both, 'raw' for non processed and 'eq' for equalized
  """
  print("Started saving chunks of data")
  corpus_len = 1003  # len(corpus)
  max_val = 0
  max_stft_len = 0
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

  memory_counter = 0

  total_time = 0
  # Get each sentence from corpus and add random noise/echos/both to the input
  # and preprocess the output. Also pad the signals to the max_val
  for i in range(corpus_len):
    start = datetime.datetime.now()
    # Original data in time domain
    data_orig_td = corpus[i].data.astype(np.float64)
    yi = pad(deepcopy(data_orig_td), max_val + NFFT//2)
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

    echosample = pad(echosample, max_val + NFFT//2)
    noisesample = pad(noisesample, max_val + NFFT//2)
    bothsample = pad(bothsample, max_val + NFFT//2)

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
    random_sample_stft = None
    if rand == 0:
      random_sample_stft = librosa.core.stft(
        echosample,
        n_fft=NFFT,
        center=True
      )
    elif rand == 1:
      random_sample_stft = librosa.core.stft(
        noisesample,
        n_fft=NFFT,
        center=True
      )
    else:
      random_sample_stft = librosa.core.stft(
        bothsample,
        n_fft=NFFT,
        center=True
      )

    max_stft_len = random_sample_stft.shape[1]
    X.append(np.abs(random_sample_stft.T))

    # print("Padded {}".format(i))
    dt = datetime.datetime.now() - start
    total_time += dt.total_seconds() * 1000
    avg_time = total_time / (i+1)
    if (i % CHUNK == CHUNK - 1):
      print("Average Time taken for {}: {}ms".format(i, avg_time))
      print("Saving temp npy file to CHUNK {}".format(memory_counter))
      # Convert to np arrays
      if is_eq or is_both:
        y_eq_temp = np.array(y_eq)

      if is_raw or is_both:
        y_raw_temp = np.array(y_raw)

      X_temp = np.array(X)

      # Save files
      np.save(
        os.path.join(
          PP_DATA_DIR,
          "model",
          "inputs_{}.npy".format(memory_counter)
        ),
        X_temp,
        allow_pickle=True
      )

      if is_eq or is_both:
        np.save(
          os.path.join(
            PP_DATA_DIR,
            "model",
            "truths_eq_{}.npy".format(memory_counter)
          ),
          y_eq_temp,
          allow_pickle=True
        )
      if is_raw or is_both:
        np.save(
          os.path.join(
            PP_DATA_DIR,
            "model",
            "truths_raw_{}.npy".format(memory_counter)
          ),
          y_raw_temp,
          allow_pickle=True
        )

      X = []
      y_eq = None
      y_raw = None

      if is_eq or is_both:
        y_eq = []
      if is_raw or is_both:
        y_raw = []

      memory_counter += 1

  if corpus_len % CHUNK > 0:
    # Convert to np arrays
    if is_eq or is_both:
      y_eq_temp = np.array(y_eq)

    if is_raw or is_both:
      y_raw_temp = np.array(y_raw)

    X_temp = np.array(X)
    end_len = len(X)

    # Save temp files
    np.save(
      os.path.join(
        PP_DATA_DIR,
        "model",
        "inputs_{}.npy".format(memory_counter)
      ),
      X_temp,
      allow_pickle=True
    )

    if is_eq or is_both:
      np.save(
        os.path.join(
          PP_DATA_DIR,
          "model",
          "truths_eq_{}.npy".format(memory_counter)
        ),
        y_eq_temp,
        allow_pickle=True
      )
    if is_raw or is_both:
      np.save(
        os.path.join(
          PP_DATA_DIR,
          "model",
          "truths_raw_{}.npy".format(memory_counter)
        ),
        y_raw_temp,
        allow_pickle=True
      )
    print("Saved blocks {}:{}".format(0, memory_counter*CHUNK + end_len))
    memory_counter += 1

  memory_counter = np.array(memory_counter)
  max_stft_len = np.array(max_stft_len)
  corpus_len = np.array(corpus_len)

  np.save(os.path.join(PP_DATA_DIR, "model", "memory_counter"), memory_counter)
  np.save(os.path.join(PP_DATA_DIR, "model", "max_stft_len"), max_stft_len)
  np.save(os.path.join(PP_DATA_DIR, "model", "corpus_len"), corpus_len)

  return memory_counter, max_stft_len, truth_type, corpus_len


def concatenate_files(truth_type=None):
  """
  Save the distributed files.
  """
  is_eq = False
  is_raw = False
  is_both = True

  if truth_type == 'eq':
    is_eq = True
    is_both = False
  elif truth_type == 'raw':
    is_raw = True
    is_both = False

  memory_counter = np.load(
    os.path.join(PP_DATA_DIR,
                 "model",
                 "memory_counter.npy")
  )
  os.remove(os.path.join(PP_DATA_DIR, "model", "memory_counter.npy"))

  max_stft_len = np.load(
    os.path.join(PP_DATA_DIR,
                 "model",
                 "max_stft_len.npy")
  )
  os.remove(os.path.join(PP_DATA_DIR, "model", "max_stft_len.npy"))

  corpus_len = np.load(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))
  os.remove(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))

  X = np.zeros(shape=(corpus_len, max_stft_len, NFFT//2 + 1))
  y_eq = None
  y_raw = None
  if is_eq or is_both:
    y_eq = np.zeros(shape=(corpus_len, max_stft_len, NFFT//2 + 1))
  if is_raw or is_both:
    y_raw = np.zeros(shape=(corpus_len, max_stft_len, NFFT//2 + 1))

  end = 0
  for file_i in range(memory_counter - 1):
    start = file_i * CHUNK
    end = start + CHUNK
    # print("Loading blocks {}:{}".format(start, end))

    X[start:end] = np.load(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "inputs_{}.npy".format(file_i))
    )
    os.remove(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "inputs_{}.npy".format(file_i))
    )
    if is_eq or is_both:
      y_eq[start:end] = np.load(
        os.path.join(PP_DATA_DIR,
                     "model",
                     "truths_eq_{}.npy".format(file_i))
      )
      os.remove(
        os.path.join(PP_DATA_DIR,
                     "model",
                     "truths_eq_{}.npy".format(file_i))
      )
    if is_raw or is_both:
      y_raw[start:end] = np.load(
        os.path.join(PP_DATA_DIR,
                     "model",
                     "truths_raw_{}.npy".format(file_i))
      )
      os.remove(
        os.path.join(PP_DATA_DIR,
                     "model",
                     "truths_raw_{}.npy".format(file_i))
      )

    print("Loaded blocks {}:{}".format(start, end))

  X[end:] = np.load(
    os.path.join(
      PP_DATA_DIR,
      "model",
      "inputs_{}.npy".format(memory_counter - 1)
    )
  )
  os.remove(
    os.path.join(
      PP_DATA_DIR,
      "model",
      "inputs_{}.npy".format(memory_counter - 1)
    )
  )
  if is_eq or is_both:
    y_eq[end:] = np.load(
      os.path.join(
        PP_DATA_DIR,
        "model",
        "truths_eq_{}.npy".format(memory_counter - 1)
      )
    )
    os.remove(
      os.path.join(
        PP_DATA_DIR,
        "model",
        "truths_eq_{}.npy".format(memory_counter - 1)
      )
    )
  if is_raw or is_both:
    y_raw[end:] = np.load(
      os.path.join(
        PP_DATA_DIR,
        "model",
        "truths_raw_{}.npy".format(memory_counter - 1)
      )
    )
    os.remove(
      os.path.join(
        PP_DATA_DIR,
        "model",
        "truths_raw_{}.npy".format(memory_counter - 1)
      )
    )

  print("Loaded blocks {}:{}".format(end, X.shape[0]))

  # Save files
  np.save(
    os.path.join(PP_DATA_DIR,
                 "model",
                 "inputs.npy"),
    X,
    allow_pickle=True
  )

  if is_eq or is_both:
    np.save(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "truths_eq.npy"),
      y_eq,
      allow_pickle=True
    )
  if is_raw or is_both:
    np.save(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "truths_raw.npy"),
      y_raw,
      allow_pickle=True
    )

  print("Saved processed dataset")

  if is_eq and not is_both:
    return X, y_eq, None
  elif is_raw and not is_both:
    return X, y_raw, None

  return X, y_raw, y_eq


def generate_dataset(truth_type):
  corpus = download_corpus()
  processed_data_path = os.path.join(PP_DATA_DIR, 'model')
  if not os.path.exists(processed_data_path):
    print("Creating preprocessed/model")
    create_preprocessed_dataset_directories()
  memory_counter, max_stft_len, truth_type, corpus_len = save_distributed_files(truth_type, corpus)
  X, y, _ = concatenate_files(truth_type)

  return X, y, _


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

  if not os.path.isfile(os.path.join(PP_DATA_DIR, 'model', 'inputs.npy')):
    X, y, _ = generate_dataset(truth_type)
  else:
    X = np.load(os.path.join(PP_DATA_DIR, 'model', 'inputs.npy'))
    y = np.load(
      os.path.join(PP_DATA_DIR,
                   'model',
                   'truths_{}.npy').format(truth_type)
    )

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


def play_sound(y, fs=NFFT):
  sounddevice.play(y, fs)
  plt.plot(y)
  plt.show()
  return


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

    X_train_norm = normalize_sample(X_train)
    X_val_norm = normalize_sample(X_val)
    y_train_norm = normalize_sample(y_train)
    y_val_norm = normalize_sample(y_val)
    model = None

    model = gen_model(X_train_norm.shape)
    print('Created Model...')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print('Compiled Model...')

    log_dir = os.path.join(
      LOGS_DIR,
      'files',
      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=1
    )

    # fit the keras model on the dataset
    cbs = [
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=100,
        verbose=1,
        mode='auto'
      ),
      tensorboard_callback
    ]

    model.fit(
      X_train_norm,
      y_train_norm,
      epochs=EPOCHS,
      batch_size=BATCH_SIZE,
      validation_data=(X_val_norm,
                       y_val_norm),
      verbose=1,
      callbacks=cbs
    )
    print('Model Fit...')

    model = save_model(model, model_name)

  min_y, max_y = np.min(y_test), np.max(y_test)
  X_test_norm = normalize_sample(X_test)
  y_test_norm = normalize_sample(y_test)

  X = X_test_norm
  y = y_test_norm

  model = model_load(model_name)

  _, mse, accuracy = model.evaluate(X, y, verbose=0)
  print('Testing accuracy: {}, Testing MSE: {}'.format(accuracy * 100, mse))

  # Predict and listen to one:
  idx = random.randint(0, len(X) - 1)
  test = X[idx]
  test = test.reshape(1, test.shape[0], test.shape[1])
  y_pred = model.predict(test)
  output_lowdim = (y_pred[0].T) * (max_y-min_y)
  output_sound = librosa.griffinlim(output_lowdim)
  # play_sound(output_sound, FS)

  return


if __name__ == '__main__':
  clear_logs()
  test_and_train(model_name='speech2speech', retrain=True)
  # print(timeit.timeit(generate_dataset, number=1))
  # generate_dataset(truth_type='raw')
