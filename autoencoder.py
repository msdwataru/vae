# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from BatchNormalization import BatchNormalization

from IPython import embed

class Autoencoder:
    def __init__(self, ch_list=[3,32,32,16,8], k_h=5, k_w=5, stddev=0.01):
        self.ch_list = ch_list
        #define learnable parameter
        with tf.variable_scope("ae"):
            #encoder
            self.w_enc1 = tf.Variable(tf.truncated_normal([28 * 28, self.ch_list[1]], stddev=stddev))
            self.b_enc1 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_enc2 = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.b_enc2 = tf.Variable(tf.zeros([self.ch_list[2]]))
            
            #decoder
            self.w_dec2 = tf.Variable(tf.truncated_normal([self.ch_list[2], self.ch_list[1]],stddev=stddev))
            self.b_dec2 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_dec1 = tf.Variable(tf.truncated_normal([self.ch_list[1], 28 * 28], stddev=stddev))
            self.b_dec1 = tf.Variable(tf.zeros([28 * 28]))

    def __call__(self, x, batch_size, train=True):
        
        #Full connection1(300)
        h_conv4 = tf.reshape(x, [-1, 1 * 28 * 28])
        inter_enc1 = tf.matmul(h_conv4, self.w_enc1) + self.b_enc1
        h_enc1 = tf.nn.tanh(inter_enc1)

        #Full connection2(20)
        inter_enc2 = tf.matmul(h_enc1, self.w_enc2) + self.b_enc2
        h_enc2 = tf.nn.tanh(inter_enc2)

        #Full connection3(300)
        inter_dec2 = tf.matmul(h_enc2, self.w_dec2) + self.b_dec2
        h_dec2 = tf.nn.tanh(inter_dec2)
        
        #Full connection4(784)
        inter_dec1 = tf.matmul(h_dec2, self.w_dec1) + self.b_dec1
        h_dec1 = tf.nn.tanh(inter_dec1)
        h_dec1 = tf.reshape(h_dec1, [-1, 28, 28, 1])
        
        return h_dec1
        
def conv2d(x, weight, batch_norm=None, train=True, activation=tf.nn.tanh):
    h_conv = tf.nn.conv2d(x, weight, strides=[1,2,2,1], padding="SAME")
    if batch_norm != None:
        h_conv = batch_norm(h_conv, train=train)
    h_conv = activation(h_conv)
    return h_conv

def deconv2d(x, weight, output_shape, batch_norm=None, train=True, activation=tf.nn.tanh):
    h_deconv = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=[1, 2, 2, 1], padding="SAME")
    if batch_norm != None:
        h_deconv = batch_norm(h_deconv, train=train)
    h_deconv = activation(h_deconv)
    return h_deconv


    
