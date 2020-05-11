import os, argparse, sys, shutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time, timeit, datetime, random

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, LSTM, TimeDistributed, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Activation, ZeroPadding2D, Flatten, Bidirectional, Conv2DTranspose, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import librosa, sounddevice

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs, MODEL_DIR, create_model_directory

import add_echoes as ae
import add_noise as an
import soundfile as sf

from copy import deepcopy

from preprocessing import process_sentence, download_corpus

root_path = 'data/'

# Global vars to change here
NFFT = 256
HOP_LENGTH = NFFT // 4
FS = 16000
BATCH_SIZE = 8
EPOCHS = 500
STACKED_FRAMES = 8
max_val = 1

CHUNK = 200

# Model Parameters
METRICS = ['mse', 'accuracy', tf.keras.metrics.RootMeanSquaredError()]
LOSS = 'mse'
OPTIMIZER = 'adam'


class BatchIdCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


def pad(data, length):
  return librosa.util.fix_length(data, length)


def normalize_sample(X):
  """ Normalize the sample """

  if X.shape[1] != NFFT//2 + 1:
    for i, x in enumerate(X):
      # axis = 0 is along column. For Conv I'm not transposing the array -> Each row is a frequence, each column is time
      # Makes sense to normalize of every time frame and not every frequency bin.
      X_norm = librosa.util.normalize(x, axis=1)
      X[i] = X_norm
  else:
    X = librosa.util.normalize(X, axis=1)

  return X


def l2_norm(vector):
  return np.square(vector)


def SDR(denoised, cleaned, eps=1e-7):  # Signal to Distortion Ratio
  a = l2_norm(denoised)
  b = l2_norm(denoised - cleaned)
  a_b = a / b
  return np.mean(10 * np.log10(a_b + eps))


def get_conv_encoder_model(input_shape=(NFFT//2 + 1, STACKED_FRAMES, 1), l2_strength=0.0):
  """
    Get the Encoder Model

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """

  # Conv2D with 32 kernels and ReLu, 3x3in time
  inputs = Input(shape=input_shape, name='encoder_input')
  x = inputs

  # -----
  x = ZeroPadding2D(((4, 4), (0, 0)))(x)
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 8],
    strides=[1,
             1],
    padding='valid',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip0 = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(skip0)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # -----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip1 = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(skip1)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = x + skip1
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(
    filters=18,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=30,
    kernel_size=[5,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = x + skip0
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(
    filters=8,
    kernel_size=[9,
                 1],
    strides=[1,
             1],
    padding='same',
    use_bias=False,
    kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
  )(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = SpatialDropout2D(0.2)(x)
  x = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same')(x)

  Encoder = tf.keras.Model(inputs=inputs, outputs=[x], name='Encoder')
  Encoder.summary()

  return Encoder


def get_encoder_model(input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)):
  """
    Get the Encoder Model
    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """

  # Conv2D with 32 kernels and ReLu, 3x3in time
  input_layer = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
  il_expand_dims = tf.expand_dims(input_layer, axis=1)
  enc_C1D_1 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_1'))(il_expand_dims)
  enc_BN_1 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_1'))(enc_C1D_1)
  enc_Act_1 = TimeDistributed(Activation("relu", name='Enc_ReLU_1'))(enc_BN_1)
  enc_C1D_2 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_2'))(enc_Act_1)
  enc_BN_2 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_2'))(enc_C1D_2)
  enc_Act_2 = TimeDistributed(Activation("relu", name='Enc_ReLU_2'))(enc_BN_2)

  # ConvLSTM1D -> Try and make this Bidirectional
  # int_input_layer = tf.reshape(tf.expand_dims(enc_Act_2, axis=1), [-1, enc_Act_2.shape[1], enc_Act_2.shape[2], 1], name='Enc_Expand_Dims')
  ConvLSTM1D = Conv2D(1, (1, 3), use_bias=False, name='Enc_ConvLSTM1D', data_format='channels_first')(enc_Act_2)
  print(ConvLSTM1D.shape)
  int_C1DLSTM_out = tf.squeeze(ConvLSTM1D, axis=[1])

  # 3 Stacked Bidirectional LSTMs
  enc_BiLSTM_1 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_1')(int_C1DLSTM_out)
  # enc_BiLSTM_2 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_2')(enc_BiLSTM_1)
  # enc_BiLSTM_3 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_3')(enc_BiLSTM_2)

  # Linear Projection into NFFT/2 and batchnorm and ReLU
  enc_Dense_1 = Dense(NFFT // 8, name='Enc_Linear_Projection')(enc_BiLSTM_1)
  enc_BN_3 = BatchNormalization(name='Enc_Batch_Norm_3')(enc_Dense_1)
  enc_Act_3 = Activation("relu", name='Enc_ReLU_3')(enc_BN_3)

  encoder = tf.keras.Model(inputs=input_layer, outputs=[enc_Act_3], name='Encoder')

  # Begin DeConvolution
  deConv_input_expand_dims = tf.reshape(tf.expand_dims(enc_Act_3, axis=1), [-1, 1, enc_Act_3.shape[1], enc_Act_3.shape[2]])
  DeC1D_filters = enc_Act_3.shape[2]
  Act = deConv_input_expand_dims
  for i in range(2):
    DeC1D = Conv2DTranspose(
      filters=DeC1D_filters * 2,
      kernel_size=(1,
                   3),
      strides=(1,
               2),
      data_format='channels_last',
      output_padding=(0,
                      1),
      padding='valid',
      name='DeConv1D_{}'.format(i + 1)
    )(Act)
    # DeC1D = TimeDistributed()
    BN = TimeDistributed(BatchNormalization(name='DeConv_Batch_norm_{}'.format(i + 1)))(DeC1D)
    Act = TimeDistributed(Activation("relu", name='DeConv_ReLU_{}'.format(i + 1)))(BN)
    DeC1D_filters *= 2

  # DeConvReshape = Conv2D(filters=1, kernel_size=(1, 1), data_format='channels_first', name='DC1D_Reshape')(Act)
  int_DeConv_out = tf.squeeze(Act, axis=[1])
  # Linear Projection into NFFT/2 and batchnorm and ReLU
  deConv_Dense_1 = Dense(NFFT//2 + 1, name='DeConv_Linear_Projection')(int_DeConv_out)
  deConv_BN_3 = BatchNormalization(name='DeConv_Batch_Norm_{}'.format(i + 1))(deConv_Dense_1)
  # deConv_Act_3 = Activation("tanh", name='DeConv_Tanh')(deConv_BN_3)
  output_layer = deConv_BN_3
  # if input_layer.shape[1] > output_layer.shape[1]:
  #   shape = [input_layer.shape[1] - output_layer.shape[1], output_layer.shape[2]]
  #   zero_padding = tf.zeros(shape, dtype=output_layer.dtype)
  #   output_layer = tf.reshape(tf.concat([output_layer, zero_padding], 1), input_layer.shape)

  ConvDeConvModel = tf.keras.Model(inputs=input_layer, outputs=[output_layer], name='ConvDeConv')
  ConvDeConvModel.summary()

  return ConvDeConvModel


def gen_model(input_shape=(NFFT//2 + 1, STACKED_FRAMES, 1)):
  """
    Define the model architecture

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """
  # Encoder = get_conv_encoder_model(input_shape)
  Encoder = get_encoder_model(input_shape)

  # model = Concatenate()([Encoder])

  return Encoder


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


def get_fvt_stft(data_stft):
  data_stft_T = []
  for sentence in data_stft:
    # sT = sentence.T
    # stacked = []
    # for idx, tm in enumerate(sT):
    #   stacked.append()
    data_stft_T.append(sentence.T)
  data_stft_T = np.array(data_stft_T)
  # if data_stft_T.shape[0] % STACKED_FRAMES != 0:
  #   zero_padding = np.zeros((STACKED_FRAMES - (data_stft_T.shape[0] % STACKED_FRAMES), data_stft_T.shape[1]), dtype=data_stft_T.dtype)
  #   data_stft_T = np.concatenate((data_stft_T, zero_padding), axis=0)

  # data_stacked_frames = []
  # i = 0
  # for i in range(data_stft_T.shape[0]):  # // STACKED_FRAMES):
  #   data_stacked_frames.append(data_stft_T[STACKED_FRAMES * i:STACKED_FRAMES * (i+1)].T)
  # data_stacked_frames = np.array(data_stacked_frames)
  return data_stft_T


def get_padded_stft(data_stft):
  padded_stft = []
  for sentence in data_stft:
    padding = np.zeros((STACKED_FRAMES, NFFT//2 + 1), dtype=data_stft.dtype)
    padded_stft.append(np.concatenate((padding, sentence), axis=0))

  padded_stft = np.array(padded_stft)
  return padded_stft


def save_distributed_files(truth_type=None, corpus=None):
  """ 
  Save input and processed files for tests
  
  @param truth_type None for both, 'raw' for non processed and 'eq' for equalized
  """
  print("Started saving chunks of data")
  corpus_len = len(corpus)
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
  pad_length = 0
  total_time = 0
  # Get each sentence from corpus and add random noise/echos/both to the input
  # and preprocess the output. Also pad the signals to the max_val
  for i in range(corpus_len):
    start = datetime.datetime.now()

    pad_length = max_val + NFFT//2
    if pad_length % STACKED_FRAMES != 0:
      pad_length += (STACKED_FRAMES - (pad_length%STACKED_FRAMES))

    # Original data in time domain
    data_orig_td = corpus[i].data.astype(np.float64)
    yi = pad(deepcopy(data_orig_td), pad_length)
    # Sampling frequency
    fs = corpus[i].fs

    # Pad transformed signals
    echosample = ae.add_echoes(data_orig_td)
    noisesample = an.add_noise(data_orig_td, sf.read(os.path.join(NOISE_DIR, "RainNoise.flac")))
    orig_sample = data_orig_td

    echosample = pad(echosample, pad_length)
    noisesample = pad(noisesample, pad_length)
    orig_sample = pad(orig_sample, pad_length)

    # Equalize data for high frequency hearing loss
    data_eq = None
    if is_eq or is_both:
      data_eq, _ = process_sentence(yi, fs=fs)
      yi_stft_eq = librosa.core.stft(data_eq, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      yi_stft_eq = librosa.util.normalize(yi_stft_eq, axis=0)
      y_eq.append(np.abs(yi_stft_eq).T)

    # Use non processed input and pad as well
    data_raw = None
    if is_raw or is_both:
      data_raw = deepcopy(yi)
      yi_stft_raw = librosa.core.stft(data_raw, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      yi_stft_raw = librosa.util.normalize(yi_stft_raw, axis=0)
      y_raw.append(np.abs(yi_stft_raw).T)

    #randomise which sample is input
    rand = random.randint(0, 1)
    random_sample_stft = None
    if rand == 0:
      random_sample_stft = librosa.core.stft(noisesample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
    else:
      random_sample_stft = librosa.core.stft(orig_sample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)

    max_stft_len = random_sample_stft.shape[1]
    random_sample_stft = librosa.util.normalize(random_sample_stft, axis=0)
    X.append(np.abs(random_sample_stft).T)

    # print("Padded {}".format(i))
    dt = datetime.datetime.now() - start
    total_time += dt.total_seconds() * 1000
    avg_time = total_time / (i+1)
    if (i % CHUNK == CHUNK - 1):
      print("Time taken for {}: {}ms".format(i, (i+1) * avg_time))
      print("Saving temp npy file to CHUNK {}".format(memory_counter))
      # Convert to np arrays
      size = 0
      if is_eq or is_both:
        y_eq_temp = np.array(y_eq)
        size += sys.getsizeof(y_eq_temp)

      if is_raw or is_both:
        y_raw_temp = np.array(y_raw)
        size += sys.getsizeof(y_raw_temp)

      X_temp = np.array(X)
      size += sys.getsizeof(X_temp)

      print("Memory used: {}".format(size / (1024*1024)))

      # Save files
      np.save(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(memory_counter)), X_temp, allow_pickle=True)

      if is_eq or is_both:
        np.save(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(memory_counter)), y_eq_temp, allow_pickle=True)
      if is_raw or is_both:
        np.save(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(memory_counter)), y_raw_temp, allow_pickle=True)

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
    np.save(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(memory_counter)), X_temp, allow_pickle=True)

    if is_eq or is_both:
      np.save(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(memory_counter)), y_eq_temp, allow_pickle=True)
    if is_raw or is_both:
      np.save(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(memory_counter)), y_raw_temp, allow_pickle=True)
    print("Saved blocks {}:{}".format(0, memory_counter*CHUNK + end_len))
    memory_counter += 1

  memory_counter = np.array(memory_counter)
  max_stft_len = np.array(max_stft_len)
  corpus_len = np.array(corpus_len)

  np.save(os.path.join(PP_DATA_DIR, "model", "memory_counter"), memory_counter)
  np.save(os.path.join(PP_DATA_DIR, "model", "max_stft_len"), max_stft_len)
  np.save(os.path.join(PP_DATA_DIR, "model", "corpus_len"), corpus_len)

  return memory_counter, max_stft_len, truth_type, corpus_len


def concatenate_files(truth_type=None, delete_flag=False):
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

  memory_counter = np.load(os.path.join(PP_DATA_DIR, "model", "memory_counter.npy"))

  max_stft_len = np.load(os.path.join(PP_DATA_DIR, "model", "max_stft_len.npy"))

  corpus_len = np.load(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))

  X = []
  y_eq = None
  y_raw = None
  if is_eq or is_both:
    y_eq = []
  if is_raw or is_both:
    y_raw = []

  end = 0
  for file_i in range(memory_counter):
    x = np.load(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(file_i)))
    if is_eq or is_both:
      y_eq_ = np.load(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(file_i)))
    if is_raw or is_both:
      y_raw_ = np.load(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(file_i)))
    for i in range(x.shape[0]):
      X.append(x[i])
      if is_eq or is_both:
        y_eq.append(y_eq_[i])
      if is_raw or is_both:
        y_raw.append(y_raw_[i])

    print("Loaded blocks {}".format(file_i))

  X = np.array(X)
  print("Loaded blocks {}:{}".format(end, X.shape[0]))

  if y_eq is None:
    y_raw = np.array(y_raw)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_raw=y_raw)
  elif y_raw is None:
    y_eq = np.array(y_eq)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_eq=y_eq)
  else:
    y_raw = np.array(y_raw)
    y_eq = np.array(y_eq)
    np.savez_compressed(os.path.join(PP_DATA_DIR, "model", "speech"), inputs=X, truths_raw=y_raw, truths_eq=y_eq)
  print("Saved speech.npz")

  if delete_flag:
    print("Deleting temp files")

    os.remove(os.path.join(PP_DATA_DIR, "model", "memory_counter.npy"))
    print("Deleted memory_counter")

    os.remove(os.path.join(PP_DATA_DIR, "model", "corpus_len.npy"))
    print("Deleted corpus_len")

    os.remove(os.path.join(PP_DATA_DIR, "model", "max_stft_len.npy"))
    print("Deleted max_stft_len")

    for file_i in range(memory_counter):
      os.remove(os.path.join(PP_DATA_DIR, "model", "inputs_{}.npy".format(file_i)))
      print("Deleted inputs_{}".format(file_i))
      if is_raw or is_both:
        os.remove(os.path.join(PP_DATA_DIR, "model", "truths_raw_{}.npy".format(file_i)))
        print("Deleted truths_raw_{}".format(file_i))
      if is_eq or is_both:
        os.remove(os.path.join(PP_DATA_DIR, "model", "truths_eq_{}.npy".format(file_i)))
        print("Deleted truths_eq_{}".format(file_i))

  if is_eq and not is_both:
    return X, y_eq, None
  elif is_raw and not is_both:
    return X, y_raw, None

  return X, y_raw, y_eq


def generate_dataset(truth_type, delete_flag=False, speaker=[]):
  corpus = download_corpus(speaker=speaker)
  processed_data_path = os.path.join(PP_DATA_DIR, 'model')
  if not os.path.exists(processed_data_path):
    print("Creating preprocessed/model")
    create_preprocessed_dataset_directories()
  memory_counter, max_stft_len, truth_type, corpus_len = save_distributed_files(truth_type, corpus)
  X, y_raw, y_eq = concatenate_files(truth_type, delete_flag)
  print("Completed generating dataset")

  if y_eq is None:
    return X, y_raw, None
  elif y_raw is None:
    return X, None, y_eq

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

  Inputs = None
  Targets_eq = None
  Targets_raw = None
  y_raw = None
  y_eq = None

  if not os.path.isfile(os.path.join(PP_DATA_DIR, 'model', 'speech.npz')):
    Inputs, Targets_raw, Targets_eq = generate_dataset(truth_type, delete_flag=True, speaker=['clb'])
  else:
    SPEECH = np.load(os.path.join(PP_DATA_DIR, 'model', 'speech.npz'))
    Inputs = SPEECH['inputs'].astype(np.float32)
    try:
      if SPEECH['truths_raw'] is not None:
        Targets_raw = SPEECH['truths_{}'.format('raw')].astype(np.float32)
    except Exception as ex:
      print(ex)
      Targets_raw

    try:
      if SPEECH['truths_eq'] is not None:
        Targets_eq = SPEECH['truths_{}'.format('eq')].astype(np.float32)
    except Exception as ex:
      print(ex)
      Targets_eq = None

  # X = np.expand_dims(get_padded_stft(Inputs), axis=[3])
  # X = get_padded_stft(Inputs)
  X = Inputs
  if Targets_raw is not None:
    y_raw = Targets_raw
  if Targets_eq is not None:
    y_eq = Targets_eq

  if truth_type == 'raw':
    y = y_raw
  elif truth_type == 'eq':
    y = y_eq

  if truth_type is None:
    # Lets start with raw
    y = y_raw

  # Reshape to [sentence, fft//2+1, 1]

  # Generate training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_model(model, model_name='speech2speech'):

  model_json = model.to_json()
  if not os.path.exists(root_path + 'models/{}'.format(model_name)):
    os.makedirs(root_path + 'models/{}'.format(model_name))
  with open(root_path + 'models/{}/model.json'.format(model_name), 'w') as json_file:
    json_file.write(model_json)

  model.save_weights(root_path + 'models/{}/model.h5'.format(model_name))

  return model


def play_sound(x, y_pred, y, fs=NFFT):
  sounddevice.play(x, fs)
  plt.plot(x)
  plt.show()

  sounddevice.play(y_pred, fs)
  plt.plot(y_pred)
  plt.show()

  sounddevice.play(y, fs)
  plt.plot(y)
  plt.show()

  return


def stride_over(data):
  """
    Stride over the dataset to maintain latency.

    @param data [sentences, samples, bins, 1]
  """
  data_out = []

  size = 0
  for idx, sentence in enumerate(data):
    data_stacked_frames = []
    for i in range(sentence.shape[0] - STACKED_FRAMES):
      data_stacked_frames.append(sentence[i:i + STACKED_FRAMES].T)
      size += sys.getsizeof(data_stacked_frames)
    data_out.append(np.array(data_stacked_frames))
    if idx % 100 == 0:
      print("Memory used: {}".format(idx * size / (1024*1024*1024)))
      size = 0

  data_out = np.reshape(np.array(data_out), [-1, NFFT//2 + 1, STACKED_FRAMES, 1])
  return data_out


def test_and_train(model_name='speech2speech', retrain=True):
  """ 
    Test and/or train on given dataset 

    @param model_name name of model to save.
    @param retrain True if retrain, False if load from pretrained model
  """

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(truth_type='eq')
  model = None

  X_train_norm = normalize_sample(X_train)
  X_val_norm = normalize_sample(X_val)
  X_test_norm = normalize_sample(X_test)

  y_train_norm = normalize_sample(y_train)
  y_val_norm = normalize_sample(y_val)
  y_test_norm = normalize_sample(y_test)

  print("X shape:", X_train_norm.shape, "y shape:", y_train_norm.shape)

  # Xtn_strided = stride_over(X_train_norm)
  # Xvn_strided = stride_over(X_val_norm)
  # Xten_strided = stride_over(X_test_norm)

  # Xtn_reshape = Xtn_strided
  # Xvn_reshape = Xvn_strided
  # Xten_reshape = Xten_strided

  # ytn_reshape = y_train_norm.reshape(-1, NFFT//2 + 1, 1, 1)
  # yvn_reshape = y_val_norm.reshape(-1, NFFT//2 + 1, 1, 1)
  # yten_reshape = y_test_norm.reshape(-1, NFFT//2 + 1, 1, 1)

  # train_dataset = tf.data.Dataset.from_tensor_slices((Xtn_reshape,
  #                                                     ytn_reshape)).batch(X_train_norm.shape[1]).shuffle(X_train.shape[0]).repeat()
  # val_dataset = tf.data.Dataset.from_tensor_slices((Xvn_reshape, yvn_reshape)).batch(X_val_norm.shape[1]).repeat(1)

  # train_dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, y_train_norm)).batch(BATCH_SIZE).shuffle(BATCH_SIZE).repeat()
  # val_dataset = tf.data.Dataset.from_tensor_slices((X_val_norm, y_val_norm)).batch(BATCH_SIZE).repeat(1)

  # print(list(train_dataset.as_numpy_iterator())[0])

  # Scale the sample X and get the scaler
  # scaler = scale_sample(X)

  # Check if model already exists and retrain is not being called again
  if (os.path.isfile(os.path.join(MODEL_DIR, model_name, 'model.json')) and not retrain):
    model = model_load()
  else:
    if not os.path.isdir(os.path.join(MODEL_DIR, model_name)):
      create_model_directory(model_name)

    baseline_val_loss = None

    model = None

    # model = gen_model(tuple(Xtn_reshape.shape[1:]))
    model = gen_model(tuple(X_train_norm.shape[1:]))
    print('Created Model...')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print('Metrics for Model...')

    # print(list(train_dataset.as_numpy_iterator())[0])

    tf.keras.utils.plot_model(model, show_shapes=True, dpi=96, to_file=os.path.join(MODEL_DIR, model_name, 'model.png'))
    print(model.metrics_names)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if (os.path.isfile(root_path + 'models/{}/model.h5'.format(model_name))):
      model.load_weights(root_path + 'models/{}/model.h5'.format(model_name))
      baseline_val_loss = model.evaluate(X_val_norm, y_val_norm)[0]
      print(baseline_val_loss)
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        baseline=baseline_val_loss
      )

    log_dir = os.path.join(LOGS_DIR, 'files', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

    model_checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      filepath=os.path.join(MODEL_DIR,
                            model_name,
                            'model.h5'),
      save_best_only=True,
      save_weights_only=False,
      mode='min'
    )

    # fit the keras model on the dataset
    cbs = [early_stopping_callback, tensorboard_callback, model_checkpoint_callback]

    model.fit(
      X_train_norm,
      y_train_norm,
      epochs=EPOCHS,
      validation_data=(X_val_norm,
                       y_val_norm),
      verbose=1,
      callbacks=cbs,
      batch_size=BATCH_SIZE
    )
    print('Model Fit...')

    model = save_model(model, model_name)

  # model = model_load(model_name)

  [loss, mse, accuracy, rmse] = model.evaluate(X_test_norm, y_test_norm, verbose=0)  # _, mse, accuracy =
  print('Testing accuracy: {}, Testing MSE: {}, Testing Loss: {}, Testing RMSE: {}'.format(accuracy * 100, mse, loss, rmse))

  # # Randomly pick 1 test
  idx = random.randint(0, len(X_test_norm) - 1)
  print(idx)
  X = X_test_norm[idx]
  y = y_test_norm[idx]
  # y = y_test_norm[idx].reshape(-1, NFFT//2 + 1)
  # min_y, max_y = np.min(y_test_norm[idx]), np.max(y_test_norm[idx])
  # min_x, max_x = np.min(y_test_norm[idx]), np.max(y_test_norm[idx])
  # print("MinY: {}\tMaxY{}".format(min_y, max_y))
  # print("MinX: {}\tMaxX{}".format(min_x, max_x))

  X = np.expand_dims(X, axis=0)
  # X = stride_over(X)

  # mean = np.mean(X)
  # std = np.std(X)
  # X = (X-mean) / std

  print(X.shape)

  # y_pred = model.predict(X)
  y_pred = np.squeeze(model.predict(X), axis=0)
  # y_pred = y_pred.reshape(-1, NFFT//2 + 1)

  print(y.shape)
  print(y_pred.shape)

  y = y.T
  y_pred = y_pred.T
  X_test_norm = X_test_norm[idx].T

  # GriffinLim Vocoder
  output_sound = librosa.core.griffinlim(y_pred)
  input_sound = librosa.core.griffinlim(X_test_norm)
  target_sound = librosa.core.griffinlim(y)

  # Play and plot all
  play_sound(input_sound, output_sound, target_sound, FS)

  return


def prepare_input_features(stft_features):
  # Phase Aware Scaling: To avoid extreme differences (more than
  # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
  noisySTFT = np.concatenate([stft_features[:, 0:STACKED_FRAMES - 1], stft_features], axis=1)
  stftSegments = np.zeros((NFFT//2 + 1, STACKED_FRAMES, noisySTFT.shape[1] - STACKED_FRAMES + 1))

  for index in range(noisySTFT.shape[1] - STACKED_FRAMES + 1):
    stftSegments[:, :, index] = noisySTFT[:, index:index + STACKED_FRAMES]
  return stftSegments


if __name__ == '__main__':
  # clear_logs()
  print("Cleared Tensorboard Logs...")
  test_and_train(model_name='ConvLSTM', retrain=True)
  # print(timeit.timeit(generate_dataset, number=1))
  # generate_dataset(truth_type='raw')
