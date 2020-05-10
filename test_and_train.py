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
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D, Flatten, Bidirectional, Conv2DTranspose
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import librosa, sounddevice

from constants import ROOT_DIR, DATA_DIR, ARCTIC_DIR, create_arctic_directory
from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs, MODEL_DIR

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
BATCH_SIZE = 4
EPOCHS = 100
max_val = 1

CHUNK = 200

# Model Parameters
METRICS = ['mse', 'accuracy', tf.keras.metrics.RootMeanSquaredError()]
LOSS = 'mse'
OPTIMIZER = 'adam'


class Conv1DTranspose(tf.keras.layers.Layer):

  def __init__(self, filters, kernel_size, strides=1, padding='valid', name=None):
    super().__init__()
    self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(filters, (kernel_size, 1), (strides, 1), padding, name=name)

  def call(self, x):
    x = tf.expand_dims(x, axis=2)
    x = self.conv2dtranspose(x)
    x = tf.squeeze(x, axis=2)
    return x


class BahdanauAttention(tf.keras.layers.Layer):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


def pad(data, length):
  return librosa.util.fix_length(data, length)


def normalize_sample(X):
  """ Normalize the sample """

  if X.shape[1] != NFFT//2 + 1:
    for i, x in enumerate(X):
      X_norm = normalize(x, axis=1, norm='max')
      X[i] = X_norm
  else:
    X = normalize(X, axis=1, norm='max')

  return X


def get_encoder_model(input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)):
  """
    Get the Encoder Model

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """

  # Conv2D with 32 kernels and ReLu, 3x3in time
  input_layer = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
  il_expand_dims = tf.reshape(tf.expand_dims(input_layer, axis=1), [-1, 1, input_layer.shape[1], input_layer.shape[2]])
  enc_C1D_1 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_1'))(il_expand_dims)
  enc_BN_1 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_1'))(enc_C1D_1)
  enc_Act_1 = TimeDistributed(Activation("relu", name='Enc_ReLU_1'))(enc_BN_1)
  enc_C1D_2 = TimeDistributed(Conv1D(filters=32, kernel_size=3, strides=2, use_bias=True, name='Enc_Conv_2'))(enc_Act_1)
  enc_BN_2 = TimeDistributed(BatchNormalization(name='Enc_Batch_Norm_2'))(enc_C1D_2)
  enc_Act_2 = TimeDistributed(Activation("relu", name='Enc_ReLU_2'))(enc_BN_2)

  # ConvLSTM1D -> Try and make this Bidirectional
  # int_input_layer = tf.reshape(tf.expand_dims(enc_Act_2, axis=1), [-1, enc_Act_2.shape[1], enc_Act_2.shape[2], 1], name='Enc_Expand_Dims')
  ConvLSTM1D = Conv2D(1, (1, 3), use_bias=False, name='Enc_ConvLSTM1D', data_format='channels_first')(enc_Act_2)
  int_C1DLSTM_out = tf.squeeze(ConvLSTM1D, axis=[1])

  # 3 Stacked Bidirectional LSTMs
  enc_BiLSTM_1 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_1')(int_C1DLSTM_out)
  enc_BiLSTM_2 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_2')(enc_BiLSTM_1)
  enc_BiLSTM_3 = Bidirectional(LSTM(NFFT // 4, return_sequences=True), name='Enc_BiLSTM_3')(enc_BiLSTM_2)

  # Linear Projection into NFFT/2 and batchnorm and ReLU
  enc_Dense_1 = Dense(NFFT // 8, name='Enc_Linear_Projection')(enc_BiLSTM_3)
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
      strides=(2,
               2),
      data_format='channels_last',
      output_padding=(1,
                      1),
      name='DeConv1D_{}'.format(i + 1)
    )(Act)
    # DeC1D = TimeDistributed()
    BN = TimeDistributed(BatchNormalization(name='DeConv_Batch_norm_{}'.format(i + 1)))(DeC1D)
    Act = TimeDistributed(Activation("relu", name='DeConv_ReLU_{}'.format(i + 1)))(BN)
    DeC1D_filters *= 2

  DeConvReshape = Conv2D(filters=1, kernel_size=(1, 1), data_format='channels_first', name='DC1D_Reshape')(Act)
  int_DeConv_out = tf.squeeze(DeConvReshape, axis=[1])
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


def get_attention_mechanism():

  attention_layer = BahdanauAttention(10)
  return


def get_decoder_model(input_shape=(BATCH_SIZE, 209, 256), frames=None):
  """
    Define the model architecture

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """
  # Define the output shape
  output_shape = (BATCH_SIZE, frames, NFFT//2 + 1)

  input_layer = tf.keras.layers.Input(shape=input_shape, name='decoder_input')

  return


def gen_model(input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)):
  """
    Define the model architecture

    @param input_shape [BATCH_SIZE, no. of frames, no. of freq bins, 1 channel]
  """
  Encoder = get_encoder_model(input_shape)

  # model = tf.keras.layers.Concatenate()([Encoder])

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
    noisesample = an.add_noise(data_orig_td, sf.read(os.path.join(NOISE_DIR, "RainNoise.flac")))
    bothsample = ae.add_echoes(noisesample)

    echosample = pad(echosample, max_val + NFFT//2)
    noisesample = pad(noisesample, max_val + NFFT//2)
    bothsample = pad(bothsample, max_val + NFFT//2)

    # Equalize data for high frequency hearing loss
    data_eq = None
    if is_eq or is_both:
      data_eq, _ = process_sentence(yi, fs=fs)
      yi_stft_eq = librosa.core.stft(data_eq, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      y_eq.append(np.abs(yi_stft_eq.T))

    # Use non processed input and pad as well
    data_raw = None
    if is_raw or is_both:
      data_raw = deepcopy(yi)
      yi_stft_raw = librosa.core.stft(data_raw, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
      y_raw.append(np.abs(yi_stft_raw.T))

    #randomise which sample is input
    rand = random.randint(0, 2)
    random_sample_stft = None
    if rand == 0:
      random_sample_stft = librosa.core.stft(echosample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
    elif rand == 1:
      random_sample_stft = librosa.core.stft(noisesample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)
    else:
      random_sample_stft = librosa.core.stft(bothsample, n_fft=NFFT, hop_length=HOP_LENGTH, center=True)

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
  X, y, _ = concatenate_files(truth_type, delete_flag)

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

  if not os.path.isfile(os.path.join(PP_DATA_DIR, 'model', 'speech.npz')):
    X, y, _ = generate_dataset(truth_type, delete_flag=True, speaker=['clb'])
  else:
    SPEECH = np.load(os.path.join(PP_DATA_DIR, 'model', 'speech.npz'))
    X = SPEECH['inputs']
    y = SPEECH['truths_{}'.format(truth_type)]

  # Generate training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05)

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
  if (os.path.isfile(root_path + 'models/{}/model.json'.format(model_name)) and not retrain):
    model = model_load()
  else:
    X = X_train
    y = y_train

    X_train_norm = normalize_sample(X_train)
    # X_train_norm = np.expand_dims(X_train_norm_, axis=0).reshape(-1, 1, X_train_norm_.shape[1], X_train_norm_.shape[2])

    X_val_norm = normalize_sample(X_val)
    # X_val_norm = np.expand_dims(X_val_norm_, axis=0).reshape(-1, 1, X_val_norm_.shape[1], X_val_norm_.shape[2])

    y_train_norm = normalize_sample(y_train)
    y_val_norm = normalize_sample(y_val)

    print("X shape:", X_train_norm.shape, "y shape:", y_train_norm.shape)

    model = None

    model = gen_model(tuple(X_train_norm.shape[1:]))
    print('Created Model...')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print('Compiled Model...')

    log_dir = os.path.join(LOGS_DIR, 'files', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, verbose=1, mode='auto')
    model_checkpoint_callback = ModelCheckpoint(
      monitor='loss',
      filepath=os.path.join(MODEL_DIR,
                            model_name,
                            'model.h5'),
      save_best_only=True,
      save_weights_only=False,
      mode='min'
    )

    # fit the keras model on the dataset
    cbs = [
      # early_stopping_callback,
      tensorboard_callback,
      model_checkpoint_callback
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

  X_test_norm = normalize_sample(X_test)
  y_test_norm = normalize_sample(y_test)

  X = X_test_norm
  y = y_test_norm

  model = model_load(model_name)

  _, mse, accuracy = model.evaluate(X, y, verbose=0)
  print('Testing accuracy: {}, Testing MSE: {}'.format(accuracy * 100, mse))

  # Randomly pick 1 test
  idx = random.randint(0, len(X) - 1)
  min_y, max_y = np.min(y_test[idx]), np.max(y_test[idx])
  min_x, max_x = np.min(y_test[idx]), np.max(y_test[idx])
  print("MinY: {}\tMaxY{}".format(min_y, max_y))
  print("MinX: {}\tMaxX{}".format(min_x, max_x))

  test = normalize_sample(X_test[idx])
  target = y_test[idx]

  # Reshape test to single sentence
  test_reshaped = np.expand_dims(test, axis=0)

  # Predict
  y_pred = np.squeeze(model.predict(test_reshaped), axis=0)

  # Rescale
  output_lowdim = (y_pred.T) * (max_y-min_y)
  input_lowdim = (test.T) * (max_x-min_x)
  target_lowdim = (target.T)  # * (max_y-min_y)

  # GriffinLim Vocoder
  output_sound = librosa.core.griffinlim(output_lowdim)
  input_sound = librosa.core.griffinlim(input_lowdim)
  target_sound = librosa.core.griffinlim(target_lowdim)

  # Play and plot all
  play_sound(input_sound, output_sound, target_sound, FS)

  return


if __name__ == '__main__':
  clear_logs()
  print("Cleared Tensorboard Logs...")
  test_and_train(model_name='speech2speech', retrain=True)
  # print(timeit.timeit(generate_dataset, number=1))
  # generate_dataset(truth_type='raw')
