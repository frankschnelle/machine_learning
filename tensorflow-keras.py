#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 10:05:48 2022

@author: frank
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import models

start = time.perf_counter()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print("Shape Trainingsdaten: {}".format(train_images.shape))
# print("Dimension Bild Nr. 1: {}".format(train_images[0].shape))
# print("Label zu Bild Nr. 1: {}".format(train_labels[0]))

# index = 0
# plttitle = "Trainingsbild Nr. {} \n Klasse: {}".format(index+1, train_labels[index])
# plt.imshow(train_images[index].reshape(28,28))
# plt.title(plttitle)
# plt.axis('off')

#Data Pre-Processing
#Ergänzung um 1 Dimension für Farbkanal (Graustufen)
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32')
train_images /= 255 #normiert die 256 Graustufen auf 0-1

#Gleiches nun auch für die Test-Bilder
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32')
test_images /= 255 #normiert die 256 Graustufen auf 0-1

# print("Shape Trainingsdaten: {}".format(train_images.shape))
# print("Dimension Bild Nr. 1: {}".format(train_images[0].shape))
# print("Label zu Bild Nr. 1: {}".format(train_labels[0]))

#Umwandlung der Bild-Label in einen zehnstelligen, binären Vektor
#korreliert mit den 10 Ausgangsneuronen für die 10 Klassen
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Gesamtzahl der Bilder
NumberTrainImages = train_images.shape[0]
NumberTestImages = test_images.shape[0]

# print(NumberTrainImages)
# print(NumberTestImages)

#Definion des CNN

#Bestimmung Format Eingabe - alternativ (28,28,1)
mnist_inputshape = train_images.shape[1:4]

#Netzwerk-Architektur
model = Sequential()

#Kodierungsblock
model.add(Conv2D(32, kernel_size=(5,5),
                 activation = 'relu',
                 input_shape = mnist_inputshape))
model.add(MaxPooling2D(pool_size=(2,2)))
#Conv_Block 2
model.add(Conv2D(64, kernel_size=(5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#Prädikationsblock
model.add(Flatten())
model.add(Dense(128, activation = 'relu', name='features'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()


#Loss- und Optimierungsfunktion
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Hyperparameter
my_batch_size = 128
my_num_classes = 10
my_epochs = 12

history = model.fit(train_images, train_labels,
                    batch_size = my_batch_size,
                    epochs = my_epochs,
                    verbose = 1,
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])

model.save('SimpleCNN_MNIST.h5')

end = time.perf_counter()
print("verstrichene Zeit: {}s".format(end - start))