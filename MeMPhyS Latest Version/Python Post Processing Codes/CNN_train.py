#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:44:40 2022

@author: Dr. Shantanu Shahane
"""
import numpy as np
import NN_functions as NN_f
import time
#tensorboard command: tensorboard --logdir logs

epochs=10; learning_rate=0.2; dropout_factor=0.0; l2_reg_lambda=0.0; n_channels=8; kernel_size=(3,3); repeat_deconv_layer=2; n_dense_neurons=25; repeat_dense_layer=3; loss_func='mse'

n_samples=100; n_pixels=64
inputs=np.random.uniform(size=(n_samples,5))
outputs=np.random.uniform(size=(n_samples,n_pixels,n_pixels,4))

cnn_model = NN_f.CNN_model_architecture_1(inputs, outputs, n_dense_neurons, repeat_dense_layer, repeat_deconv_layer, kernel_size, dropout_factor, l2_reg_lambda)
cnn_model.summary()
NAME = "CNN_"+str(int(time.time()))
model, tensorboard, csv_logger, checkpoint, checkpoint_best_val_accuracy, checkpoint_best_train_accuracy, checkpoint_best_val_loss, checkpoint_best_train_loss = NN_f.NN_model_settings(cnn_model, NAME, learning_rate, loss_func)

cnn_model.fit(inputs, outputs, batch_size=50, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=[ csv_logger, checkpoint_best_val_accuracy, checkpoint_best_train_accuracy, checkpoint_best_val_loss, checkpoint_best_train_loss, tensorboard], verbose=1)
history=np.loadtxt('CNN_csvlog.csv', delimiter=',', skiprows=1); NN_f.plot_history(history, 'CNN', 'semilogy');