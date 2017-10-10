#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from glob import glob

files = glob('checkpoints/*')
files.sort()
weights = None
if len(files) > 0:
    weights = files[-1]
    print(weights)
filename = 'lyrik.txt'
raw_text = open(filename, encoding='utf8').read()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

print("Characters: %d (different: %d)" % (len(raw_text), len(chars)))

seq_length = 100
dataX = []
dataY = []

for i in range(len(raw_text) - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

X = np.reshape(dataX, (len(dataX), seq_length, 1))/float(len(chars))
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),
          return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
if weights is not None:
    model.load_weights(weights)
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = 'checkpoints/{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0,
                             save_best_only=True, mode='min')
filepath_last = 'checkpoints/last.hdf5'
checkpoint_last = ModelCheckpoint(filepath_last, monitor='loss', verbose=0,
                                  save_best_only=True, mode='min')
callback_list = [checkpoint, checkpoint_last]
model.fit(X, Y, epochs=20, batch_size=256, callbacks=callback_list)
