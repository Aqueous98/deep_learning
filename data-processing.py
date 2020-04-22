import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice
import numpy as np
import add-echoes as ae
import add-noise as an
import preprocessing as pp
import random
import tensorflow as tf

from tensorflow.keras import layers

corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl', 'slt'])

#the two outermost layers need to be dense with the dimensionality of the input/output
model = tf.keras.Sequential()
model.add(layers.Dense(50))
model.add(layers.LSTM(64))
model.add(layers.Dense(50))

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
    targets[i] = pp.process_sentence(sample, 16000)
    
    #randomise which sample is input
    input = 0
    rand = random.randint(0, 3)
    if rand==0:
        samples[i] = sample
    elif rand==1:
        samples[i] = echosample
    elif rand==2:
        #add just noise
        samples[i] = sample
    else:
        #add both
        samples[i] = sample
        
#train the network
model.fit(samples, targets, epochs = 50)

#test
input = corpus[len(corpus)-1]
sounddevice.play(input, 16000)
plt.figure()
plt.plot(input)
plt.show()

#run the network on the test sample
output = model.predict(input)
#plot and play output
sounddevice.play(output, 16000)
plt.plot(output)
plt.show()
