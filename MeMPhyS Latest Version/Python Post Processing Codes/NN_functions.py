#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:44:21 2022

@author: Dr. Shantanu Shahane
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import tensorflow.keras.backend as tf_back
import matplotlib.pyplot as plt

def CNN_model_architecture_1(inputs, outputs, n_dense_neurons, repeat_dense_layer, repeat_deconv_layer, kernel_size, dropout_factor, l2_reg_lambda, reshape_size=(4,4,4)):
    assert outputs.shape[1]==outputs.shape[2]
    assert (outputs.shape[1]==2**5) or (outputs.shape[1]==2**6) or (outputs.shape[1]==2**7) or (outputs.shape[1]==2**8) or (outputs.shape[1]==2**9) or (outputs.shape[1]==2**10)
    assert reshape_size[1]==reshape_size[2]
    assert (reshape_size[1]==4) or (reshape_size[1]==8) or (reshape_size[1]==16)
    # n_pixels=outputs.shape[1]; n_channels=outputs.shape[-1];
    n_channels_0 = outputs.shape[-1]*int(outputs.shape[1]/reshape_size[1])

    model = Sequential();

    model.add(Dense(n_dense_neurons, activation='relu', input_shape=(inputs.shape[1],), kernel_regularizer=regularizers.l2(l2_reg_lambda)));
    model.add(Dropout(dropout_factor)); model.add(BatchNormalization())
    for i in range(repeat_dense_layer-1):
        model.add(Dense(n_dense_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg_lambda)));
        model.add(Dropout(dropout_factor)); model.add(BatchNormalization())

    model.add(Dense(np.prod(reshape_size), activation='relu', kernel_regularizer=regularizers.l2(l2_reg_lambda)));
    model.add(Dropout(dropout_factor)); model.add(BatchNormalization())
    model.add(Reshape(reshape_size));

    # model.add(Conv2DTranspose(n_channels_0, kernel_size, strides=(1, 1), padding='same', activation = 'relu'));
    # print(model.output_shape)

    while (model.output_shape[1]<outputs.shape[1]):
        n_channels_0=int(n_channels_0/2)
        model.add(Conv2DTranspose(n_channels_0, kernel_size, strides=(2, 2), padding='same', activation = 'relu'));
        model.add(Dropout(dropout_factor)); model.add(BatchNormalization())
        for i_deconv in range(repeat_deconv_layer-1):
            model.add(Conv2DTranspose(n_channels_0, kernel_size, strides=(1, 1), padding='same', activation = 'relu'));
            model.add(Dropout(dropout_factor)); model.add(BatchNormalization())
        # print(model.output_shape)

    model.add(Conv2DTranspose(n_channels_0, kernel_size, strides=(1, 1), padding='same', activation = 'linear'))
    assert model.output_shape[1:] == outputs.shape[1:]
    return model

# def tfa_metrics_r_square_wrapper(y_true, y_pred):
#     metric = tfa.metrics.r_square.RSquare()
#     metric.update_state(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]))
#     result = metric.result()
#     return result

def coeff_determination_tf(y_true, y_pred):
    SS_tot = tf_back.sum( tf_back.square( tf.reshape(y_true, [-1]) - tf_back.mean(tf.reshape(y_true, [-1])) ) )
    SS_res = tf_back.sum( tf_back.square( tf.reshape(y_true, [-1]) - tf.reshape(y_pred, [-1]) ) )
    R_square = 1.0 - (SS_res/SS_tot)
    return R_square

def coeff_determination_np(y_true, y_pred):
    #reference: https://en.wikipedia.org/wiki/Coefficient_of_determination
    SS_tot = np.sum( (y_true - np.mean(y_true))**2 )
    SS_res = np.sum( (y_true - y_pred)**2 )
    R_square = 1.0 - (SS_res/SS_tot)
    return R_square

def NN_model_settings(model, NAME, learning_rate, loss_func):
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    csv_logger=CSVLogger('CNN_csvlog.csv', separator=',', append=False)
    filepath = "CNN-{epoch:d}-{val_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto', save_freq='epoch')
    checkpoint_best_val_accuracy = ModelCheckpoint('CNN_best_val_accuracy.hdf5', monitor='val_coeff_determination_tf', verbose=1, save_best_only=True, mode='max', save_freq='epoch')
    checkpoint_best_train_accuracy = ModelCheckpoint('CNN_best_train_accuracy.hdf5', monitor='coeff_determination_tf', verbose=1, save_best_only=True, mode='max', save_freq='epoch')
    checkpoint_best_val_loss = ModelCheckpoint('CNN_best_val_loss.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')
    checkpoint_best_train_loss = ModelCheckpoint('CNN_best_train_loss.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

#    optimizer1 = optimizers.SGD(lr=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
#    optimizer1 = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
#    optimizer1 = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer1 = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#    optimizer1 = optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.5, nesterov=True)
#    optimizer1 = optimizers.Adadelta(learning_rate=1.0, rho=0.95) #https://keras.io/optimizers/
    model.compile(loss=loss_func, optimizer=optimizer1, metrics=[coeff_determination_tf]) #tfa.metrics.r_square.RSquare()
    return model, tensorboard, csv_logger, checkpoint, checkpoint_best_val_accuracy, checkpoint_best_train_accuracy, checkpoint_best_val_loss, checkpoint_best_train_loss

def plot_history(history, fname, loss_plot_type):
    plt.figure(); plt.xlabel('Epoch'); plt.ylabel('Accuracy');
    plt.plot(history[:,0],history[:,1],'g', label='Train');
    plt.plot(history[:,0],history[:,3],'r', label='Validation');
    plt.legend(); plt.savefig(fname+'_accuracy.png')

    plt.figure(); plt.xlabel('Epoch'); plt.ylabel('Loss');
    if loss_plot_type == 'semilogy':
        plt.semilogy(history[:,0],history[:,2], 'g', label='Train');
        plt.semilogy(history[:,0],history[:,4], 'r',label='Validation');
    else:
        plt.plot(history[:,0],history[:,2], 'g', label='Train');
        plt.plot(history[:,0],history[:,4], 'r',label='Validation');
    plt.legend(); plt.savefig(fname+'_loss.png')