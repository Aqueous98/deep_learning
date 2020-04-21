import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice
import numpy as np
import add-echoes as ae
import add-noise as an
import preprocessing as pp
import random

corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl', 'slt'])

#create network here

for i in len(corpus)*(3/4):
    sample = corpus[i].data
    echosample = ae.add_echoes(sample)
    #add noise here
    #add echoes and noise here
    target = pp.process_sentence(sample, 16000)
    
    #randomise which sample is input
    input = 0
    rand = random.randint(0, 3)
    if rand==0:
        input = sample
    elif rand==1:
        input = echosample
    elif rand==2:
        #add just noise
        input = sample
    else:
        #add both
        input = sample
    #train on this sample, maybe spectrogram first
    
#test
input = corpus[len(corpus)-1]
input.play()
plt.figure()
input.plot()
plt.show()

#run the network on the test sample
#plot and play output
