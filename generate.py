#!/usr/bin/env python3

import numpy as np
from glob import glob
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

files = glob('checkpoints/*')
files.sort()
weights = files[-1]
print(weights)
filename = 'lyrik.txt'
raw_text = open(filename, encoding='utf8').read()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print("Characters: %d (different: %d)" % (len(raw_text), len(chars)))

seq_length = 100
dataX = []
dataY = []

for i in range(len(raw_text) - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
print(len(dataX))

X = np.reshape(dataX, (len(dataX), seq_length, 1))/float(len(chars))
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),
          return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.load_weights(weights)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
for x in pattern:
    print(int_to_char[x], end='')
print('')
print('---------------')
line_count = 0
while line_count < 10:
    X = np.reshape(pattern, (1, len(pattern), 1))/float(len(chars))
    prediction = model.predict(X, verbose=0)
    s = 0
    selected = False
    index = 0
    while not selected:
        r = np.random.ranf()
        i = np.random.randint(0, len(prediction[0]))
        if r < prediction[0][i]:
            selected = True
            index = i
    result = int_to_char[index]
    if result == '\n':
        line_count += 1
    print(result, end='')
    sys.stdout.flush()
    pattern.append(index)
    pattern = pattern[1:]
print('')
