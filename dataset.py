import os

import time
import datetime

import numpy as np

import random

import argparse, sys, shutil
from pathlib import Path

import pyroomacoustics as pra
from pyroomacoustics.datasets import Sample, Dataset, CMUArcticCorpus
from pyroomacoustics.datasets import CMUArcticSentence

import librosa

from sklearn.model_selection import train_test_split

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs

import add_echoes as ae
import add_noise as an
import soundfile as sf

from copy import deepcopy

from preprocessing import process_sentence, download_corpus

# Global vars to change here
NFFT = 512
FS = 16000

CHUNK = 50


def pad(data, length):
  return librosa.util.fix_length(data, length)


def generate_dataset(truth_type=None, speaker=[], corpus_len=None):
  """ 
  Save input and processed files for tests
  
  @param truth_type None for both, 'raw' for non processed and 'eq' for equalized
  @param speaker list of speakers
  @param corpus_len provide a lenth. If None the entire corpus will be parsed.
  """

  corpus = None

  if len(speaker) == 0:
    print("All speakers are being selected")
    speaker = None
    corpus = download_corpus()
  else:
    print("Speakers selected are {}".format(speaker))
    corpus = download_corpus(speaker=speaker)

  processed_data_path = os.path.join(PP_DATA_DIR, 'model')
  if not os.path.exists(processed_data_path):
    print("Creating preprocessed/model directory")
    create_preprocessed_dataset_directories()

  if corpus_len is None:
    print("This will run on the entire corpus")
    corpus_len = len(corpus)
  else:
    print("Corpus length is for {}".format(corpus_len))

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

    #randomise which sample is input
    rand = random.randint(0, 2)
    random_sample = None
    if rand == 0:
      random_sample = pad(echosample, max_val + NFFT//2)
    elif rand == 1:
      random_sample = pad(noisesample, max_val + NFFT//2)
    else:
      random_sample = pad(bothsample, max_val + NFFT//2)

    # Append random_sample to input list
    X.append(random_sample)

    # Equalize data for high frequency hearing loss
    data_eq = None
    if is_eq or is_both:
      data_eq, _ = process_sentence(yi, fs=fs)
      y_eq.append(data_eq)

    # Use non processed input and pad as well
    data_raw = None
    if is_raw or is_both:
      data_raw = deepcopy(yi)
      y_raw.append(data_raw)

    # print("Padded {}".format(i))
    dt = datetime.datetime.now() - start
    total_time += dt.total_seconds() * 1000
    avg_time = total_time / (i+1)
    if (i % CHUNK == CHUNK - 1):
      print("Time taken for {}: {}ms".format(i, (i+1) * avg_time))

  X = np.array(X)

  # Save the data
  np.save(os.path.join(PP_DATA_DIR, "model", "inputs.npy"), X)

  if is_eq or is_both:
    y_eq = np.array(y_eq)
    np.save(os.path.join(PP_DATA_DIR, "model", "truths_eq.npy"), y_eq)

  if is_raw or is_both:
    y_raw = np.array(y_raw)
    np.save(os.path.join(PP_DATA_DIR, "model", "truths_raw.npy"), y_eq)

  if y_raw is not None and y_eq is None:
    np.savez_compressed(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "speech"),
      inputs=X,
      truths_raw=y_raw
    )
  elif y_eq is not None and y_raw is None:
    np.savez_compressed(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "speech"),
      inputs=X,
      truths_raw=y_eq
    )
  else:
    np.savez_compressed(
      os.path.join(PP_DATA_DIR,
                   "model",
                   "speech"),
      inputs=X,
      truths_raw=y_raw,
      truths_eq=y_eq
    )

  if is_eq and not is_both:
    return X, y_eq, None
  elif is_raw and not is_both:
    return X, y_raw, None

  return X, y_raw, y_eq


def load_dataset(truth_type='raw', speaker=None, corpus_len=None):
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

  if not os.path.isfile(os.path.join(PP_DATA_DIR, 'model', 'speech.npz')):
    X, y, _ = generate_dataset(truth_type, speaker, corpus_len)
  else:
    SPEECH = np.load(os.path.join(PP_DATA_DIR, 'model', 'speech.npz'))
    X = SPEECH['inputs']
    y = SPEECH['truths_{}'.format(truth_type)]

  # Generate training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test():
  generate_dataset('raw', corpus_len=200)


if __name__ == '__main__':
  test()