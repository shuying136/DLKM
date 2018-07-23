# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence,text
from keras.datasets import imdb
from attention import*
from keras.models import Model
from keras.layers import *
from keras import regularizers
from model import*
from utils_ import load_data,load_data_small
import keras
from keras import optimizers
import tensorflow as tf
X_train,y_train,X_test,y_test = load_data()
'''
#yelp_review_polarity_csv
max_features = 100000
maxlen = 200
batch_size = 500
num_classes = 2

#DBPendia 
max_features = 400000
maxlen = 80
batch_size = 500
num_classes = 14

#ag_news_csv
max_features = 500000
maxlen = 200
batch_size = 500
num_classes = 10
'''
#yahoo_answers_csv
max_features = 80000   
maxlen = 50
num_classes = 4
batch_size=500
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(len(tokenizer.word_counts))

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
name = 'lstm'
with tf.device('/cpu:0'):
    model = build_lstm()
    model.summary()

# try using different optimizers and different optimizer configs
adam = optimizers.Adam(lr=0.001)
gadient = optimizers.SGD(lr=0.05)
model_gpu = keras.utils.multi_gpu_model(model, gpus=2)
model_gpu.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

print('Train {} ...'.format(name))
model_gpu.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=(x_test, y_test))
