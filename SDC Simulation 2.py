# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:25:20 2020

@author: karti
"""


import pandas
import numpy as np
from keras.preprocessing import image
from keras.layers import Conv2D,Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
import scipy.misc
# import model
from subprocess import call
import os
import cv2

import tensorflow as tf
'''
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(48, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron.
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape = (224,224,3) ),
tf.keras.layers.Conv2D(24, (5, 5), activation = 'elu', strides = (2, 2) ),
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (2, 2) ),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (2, 2) ),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(100, activation='elu'),
tf.keras.layers.Dense(50, activation='elu'),
tf.keras.layers.Dense(10, activation='elu'),
tf.keras.layers.Dense(1, activation = 'linear')
])

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='selu', input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(48, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron.
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape = (224,224,3) ),
tf.keras.layers.Conv2D(24, (5, 5), activation = 'elu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),

tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu',  kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dense(50, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dense(10, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dense(1, activation = 'linear', kernel_regularizer=regularizers.l2(0.001))
])
model.summary()

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(48, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron.
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(24, (5, 5), activation = 'elu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001), input_shape = (224, 224,3) ),
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),

tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu',  kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear', kernel_regularizer=regularizers.l2(0.001))
])

import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import Model
# load model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# example of loading the inception v3 model
from keras.applications.inception_v3 import InceptionV3
# load model
model = InceptionV3(include_top=False, input_shape=(224, 224, 3))

for layer in model.layers:
  layer.trainable = False

y = model.output
y = tensorflow.keras.layers.Flatten()(y)
y = tensorflow.keras.layers.Dropout(0.5)(y)
y = tensorflow.keras.layers.Dense(128, activation = 'relu')(y)
y = tensorflow.keras.layers.Dropout(0.2)(y)
y = tensorflow.keras.layers.Dense(64, activation = 'relu')(y)
y = tensorflow.keras.layers.Dropout(0.2)(y)
y = tensorflow.keras.layers.Dense(1, activation='linear')(y)
model = Model(inputs=model.input, outputs=y)

model = Sequential()
model.add(Conv2D(24,[5,5],activation='relu',input_shape=[224,224,3],strides=[2,2],kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(36,[5,5],activation='relu',strides = [2,2],kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(48,[5,5],activation='relu',strides = [2,2],kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64,[3,3],activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64,[3,3],activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(24,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear',kernel_regularizer=regularizers.l2(0.001)))

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(100, 100, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(512,(3,3), activation = 'relu'),# kernel_regularization = regularizers.l2(0.001)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    
                                    tf.keras.layers.Flatten(),
                                    
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(256, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1, activation='linear')
])

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(100, 100, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(512,(3,3), activation = 'relu'),# kernel_regularization = regularizers.l2(0.001)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    
                                    tf.keras.layers.Flatten(),
                                    
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(256, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(64, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1, activation='linear')
])
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(48, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron.
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='selu', input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(48, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron.
    tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(24, (5, 5), activation = 'selu', strides = (2, 2), input_shape = (224,224,3), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(36, (5, 5), activation = 'selu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'selu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'selu', kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'selu', kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(64, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(32, activation = 'selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear',kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(36, (5, 5), activation = 'selu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001), input_shape = (224,224,3) ),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'selu', strides = (2, 2), kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'selu', kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'selu', kernel_regularizer=regularizers.l2(0.001) ),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='selu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear',kernel_regularizer=regularizers.l2(0.001))
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), input_shape = (224,224,3) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (1, 1), ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256, (3, 3), activation = 'elu', ),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear')
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), input_shape = (224,224,3) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (1, 1), ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256, (3, 3), activation = 'elu', ),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(256, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear')
])

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), input_shape = (224,224,3) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (1, 1), ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(1024, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(256, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear')
])
'''
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(36, (5, 5), activation = 'elu', strides = (1, 1), input_shape = (224,224,3) ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(48, (5, 5), activation = 'elu', strides = (1, 1), ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3, 3), activation = 'elu' ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128, (3, 3), activation = 'elu', ),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256, (3, 3), activation = 'elu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(1024, (3, 3), activation = 'elu', kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(256, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='elu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'linear')
])
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])
model.summary()
model.load_weights('C:/Users/karti/Downloads/model-08-0.13.h5')
#model.load_weights('C:/Users/karti/Downloads/model-09-0.16.h5')
#model.load_weights('C:/Users/karti/Downloads/model-06-0.17.h5')

img_str = cv2.imread('C:/Users/karti/OneDrive/Desktop/steering_wheel_image.jpg',0)
rows,cols = img_str.shape

smoothed_angle = 0

i = 30000
while(cv2.waitKey(10) != ord('q') and i<31300):
	img1 = image.load_img("C:/Users/karti/OneDrive/Desktop/07012018/data/" + str(i) + ".jpg",color_mode='rgb')
	img1 = image.image.img_to_array(img1)/255.0
	img = image.load_img("C:/Users/karti/OneDrive/Desktop/07012018/data/" + str(i) + ".jpg",color_mode='rgb',target_size=[224, 224])
	img = image.img_to_array(img)/255.0
	img_resh = np.reshape(img,[1,224,224,3])
	degrees = np.squeeze(model.predict(img_resh) * 180.0 / scipy.pi)
	print("Predicted steering angle: " + str(degrees) + " degrees" + str(i))
	cv2.imshow("frame", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
	smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),(-1*smoothed_angle), 1)
	dst = cv2.warpAffine(img_str,M,(cols,rows))
	cv2.imshow("steering wheel", dst)
	i += 1

cv2.destroyAllWindows()
