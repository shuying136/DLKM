# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from attention import*
from keras.models import Model
from keras.layers import*
from keras.models import Sequential
from Batch_LSTM import BatchNormLSTM
from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints
import numpy as np
from keras.layers import Concatenate, Dot, Merge, Multiply, RepeatVector
from keras.engine import  InputSpec
import tensorflow as tf
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, ZeroPadding1D
from keras.layers import  Activation,GlobalAveragePooling1D,BatchNormalization
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, CuDNNGRU, Flatten, SpatialDropout1D
'''
#yelp_review_polarity_csv
gru_len = 256
max_features = 100000    
maxlen = 200
batch_size = 500
num_classes = 2
embed_size = 300
#0.9591

#DBPedia data
gru_len = 128
max_features = 400000    
maxlen = 80
batch_size = 128
num_classes = 14
embed_size = 300
'''
#ag_news_csv
gru_len = 100
dense_layer = 100
max_features = 80000    
maxlen = 50
num_classes = 4
embed_size = 300
'''

#amazon_review_full_csv
gru_len = 256
max_features = 50000    
maxlen = 50
num_classes = 4
embed_size = 300

#yahoo_answers_csv
gru_len = 496
max_features = 500000    
maxlen = 200
num_classes = 10
embed_size = 300
'''
class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        
        # return flattened output
        return Flatten()(top_k)


class KMaxPooling_non_flatten(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling_non_flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis

        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)

    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]

        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)

        return transposed_back


class Folding(Layer):

    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))

    def call(self, x):
        input_shape = x.get_shape().as_list()

        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors
        splits = tf.split(x, num_or_size_splits=int(input_shape[2]/2), axis=2)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=2) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=2)
        return row_reduced
        
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
def build_dcnn():
    model_1 = Sequential([
    Embedding(max_features, embed_size),
    ZeroPadding1D((49,49)),
    Conv1D(64, 50, padding="same"),
    KMaxPooling_non_flatten(k=5, axis=1),
    Activation("relu"),
    ZeroPadding1D((24,24)),
    Conv1D(64, 25, padding="same"),
    Folding(),
    KMaxPooling_non_flatten(k=5, axis=1),
    Activation("relu"),
    Flatten(),
    Dense(num_classes, activation="softmax")
    ])
    return model_1
def build_cnn_lstm():
    maxlen = 100
    embedding_size = 128
    max_features = 20000
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4
    
    # LSTM
    lstm_output_size = 70
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
def build_fasttext():
    S_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings = Embedding(max_features, embed_size)(S_inputs)
    avg_pool = GlobalAveragePooling1D()(embeddings)
    outputs = Dense(num_classes, activation='softmax')(avg_pool)
    
    model = Model(inputs=S_inputs, outputs=outputs)
    return model
def build_cnn():
    S_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings = Embedding(max_features, embed_size)(S_inputs)
    filter_sizes = [3,4,5]
    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='relu')(embeddings)
        l_pool = MaxPool1D(filter_size)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
    # since the text is too long we are maxpooling over 100
    # and not GlobalMaxPool1D
    print(l_cov1)
    l_pool1 = MaxPool1D(10)(l_cov1)
    l_flat = Flatten()(l_pool1)
    l_dense = Dense(128, activation='relu')(l_flat)
    l_out = Dense(num_classes, activation='softmax')(l_dense)
    model_1 = Model(inputs=[S_inputs], outputs=l_out)
    return model_1

def build_lstm(): 
    S_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings = Embedding(max_features, embed_size)(S_inputs)
    #embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
    O_seq = SpatialDropout1D(0.1)(embeddings)
    #O_seq = Dropout(0.1)(embeddings)
    O_seq= Bidirectional(CuDNNGRU(gru_len,return_sequences=True))(O_seq)
    #O_seq = Bidirectional(BatchNormLSTM(gru_len,return_sequences=True))(O_seq)
    #O_seq = BatchNormLSTM(128,return_sequences=True)(O_seq)
    O_seq = KMaxPooling(k=5)(O_seq)
    #O_seq = LSTM(128,dropout=0.5,recurrent_dropout=0.5)(embeddings)
    #O_seq = Conv1D(512,2)(O_seq)
    O_seq = BatchNormalization()(O_seq)
    #O_seq_0 = GlobalMaxPooling1D()(O_seq)
    #O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
    #O_seq_1 = GlobalAveragePooling1D()(O_seq)
    #conc = concatenate([O_seq_0, O_seq_1])
    #O_seq = Dropout(0.2)(O_seq)
    O_seq = Dense(dense_layer,activation='relu')(O_seq)
    O_seq = BatchNormalization()(O_seq)
    outputs = Dense(num_classes, activation='softmax')(O_seq)
    
    model = Model(inputs=S_inputs, outputs=outputs)
    return model
def build_batchnorm_lstm(): 
    S_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings = Embedding(max_features, embed_size)(S_inputs)
    #embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
    O_seq = SpatialDropout1D(0.1)(embeddings)
    #O_seq= Bidirectional(CuDNNGRU(gru_len,return_sequences=True))(O_seq)
    O_seq = Bidirectional(BatchNormLSTM(gru_len,return_sequences=True))(O_seq)
    O_seq = BatchNormalization()(O_seq)
    O_seq = KMaxPooling(k=1)(O_seq)
    #O_seq = LSTM(128,dropout=0.5,recurrent_dropout=0.5)(embeddings)
    #O_seq = Conv1D(512,2)(O_seq)
    
    #O_seq_0 = GlobalMaxPooling1D()(O_seq)
    #O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
    #O_seq_1 = GlobalAveragePooling1D()(O_seq)
    #conc = concatenate([O_seq_0, O_seq_1])
    #O_seq_2 = Dropout(0.5)(O_seq)
    #O_seq = Dense(128,activation='relu')(O_seq)
    #O_seq = BatchNormalization()(O_seq)
    outputs = Dense(num_classes, activation='softmax')(O_seq)
    
    model = Model(inputs=S_inputs, outputs=outputs)
    return model
if __name__ == '__main__':
    model = build_dcnn()
    model.summary()
    
    