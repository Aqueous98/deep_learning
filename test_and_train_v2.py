import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, TimeDistributed, Bidirectional
import librosa
import random
import sounddevice
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from constants import create_preprocessed_dataset_directories, PP_DATA_DIR
from constants import NOISE_DIR, LOGS_DIR, clear_logs

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def load_dataset(rstart, rstop, truth_type='raw'):
	X = []
	y = []
	X_train = None
	X_test = None
	X_val = None
	y_test = None
	y_train = None
	y_val = None

	for i in range(rstart, rstop):
		if i%100==0:
			print("Load:",i)
		SPEECH = np.load(os.path.join(PP_DATA_DIR, 'model', str('speech'+str(i)+'.npz') ))
		X.append( np.abs(librosa.core.stft(SPEECH['inputs'],n_fft=512).T ) )
		y.append( np.abs(librosa.core.stft(SPEECH['truths_{}'.format(truth_type)],n_fft=512).T ) )

  # Generate training and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # Generate validation set
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

	return (X_train, y_train), (X_val, y_val), (X_test, y_test)
	
#Define Variables
max_val = 1
NFFT = 512
BATCH_SIZE = 32
input_shape=(BATCH_SIZE, max_val, NFFT//2 + 1)	
output_shape = input_shape[2]
LOSS = 'mse'
OPTIMIZER = 'adam'
METRICS = ['mse', 'accuracy']


#Define Model
model = Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100)))
model.add(Bidirectional(LSTM(200, return_sequences=True, activation = "sigmoid")))
#model.add(Bidirectional(LSTM(150)))
#model.add(Bidirectional(LSTM(50)))
#model.add(LSTM(100))
#model.add(LSTM(200))
model.add(Dense(output_shape))
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

log_dir = os.path.join(LOGS_DIR, 'files', datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1 )
cbs = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=1,
        mode='auto'), tensorboard_callback]

dataset_size = 2000
BatchesSizes = 1000
EPOCHS = 2
iterations = dataset_size/BatchesSizes
  
for i in range(2): #iterations
	rstart = BatchesSizes*(i)
	rstop = rstart+BatchesSizes

	(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(rstart, rstop)
	
	#normalize
	X_train_norm = np.array(X_train / np.linalg.norm(X_train) )
	X_val_norm = np.array( X_val / np.linalg.norm(X_val) )
	y_train_norm = np.array( y_train / np.linalg.norm(y_train) )
	y_val_norm = np.array( y_val / np.linalg.norm(y_val) )
	
	print(X_train_norm.shape)
	print(X_val_norm.shape)
	print(y_val_norm.shape)
	
	model.fit(X_train_norm, y_train_norm, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_norm,
      y_val_norm), verbose=1, callbacks=cbs)
	
idx = random.randint(0, len(X_train_norm) - 1)	
test = X_train_norm[idx]
test = test.reshape(1, test.shape[0], test.shape[1])
y_pred = model.predict(test)
min_y, max_y = np.min(y_test), np.max(y_test)
output_lowdim = (y_pred[0].T) * (max_y-min_y)
output_sound = librosa.griffinlim(output_lowdim)
sounddevice.play(output_sound, 16000)
plt.plot(y)
plt.show()

print("finished")












  
  
  

