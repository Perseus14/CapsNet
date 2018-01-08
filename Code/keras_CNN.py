import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

import feature_extractor

[[X_train, Y_train, X_val, Y_val, X_test, Y_test],labels] = feature_extractor.load_data()

num_labels = len(labels)
filter_size = 2

Y_train = np.eye(num_labels)[Y_train]
Y_val = np.eye(num_labels)[Y_val]
Y_test = np.eye(num_labels)[Y_test]

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
