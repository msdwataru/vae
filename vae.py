# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from BatchNormalization import BatchNormalization

from IPython import embed

class VAE:
    def __init__(self, ch_list=[3,32,32,16,8], k_h=5, k_w=5, stddev=0.01):
        self.ch_list = ch_list
        #define learnable parameter
        with tf.variable_scope("ae"):
            #encoder
            self.w_enc1 = tf.Variable(tf.truncated_normal([28 * 28, self.ch_list[1]], stddev=stddev))
            self.b_enc1 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_enc2_mu = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.b_enc2_mu = tf.Variable(tf.zeros([self.ch_list[2]]))
            self.w_enc2_ln_var = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.b_enc2_ln_var = tf.Variable(tf.zeros([self.ch_list[2]]))
            
            #decoder
            self.w_dec2 = tf.Variable(tf.truncated_normal([self.ch_list[2], self.ch_list[1]],stddev=stddev))
            self.b_dec2 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_dec1 = tf.Variable(tf.truncated_normal([self.ch_list[1], 28 * 28], stddev=stddev))
            self.b_dec1 = tf.Variable(tf.zeros([28 * 28]))

    def __call__(self, x, batch_size, train=True):

        #Encoder1 (300)
        inputs = tf.reshape(x, [-1, 1 * 28 * 28])
        inter_enc1 = tf.matmul(inputs, self.w_enc1) + self.b_enc1
        h_enc1 = tf.nn.tanh(inter_enc1)

        #Encoder2 mu (20)
        inter_enc2_mu = tf.matmul(h_enc1, self.w_enc2_mu) + self.b_enc2_mu
        #h_enc2_mu = tf.nn.tanh(inter_enc2)

        #Encoder2 ln_var (20)
        inter_enc2_ln_var = tf.matmul(h_enc1, self.w_enc2_ln_var) + self.b_enc2_ln_var
        #h_enc2 = tf.nn.tanh(inter_enc2)

        #Sample z
        z = sample_z(inter_enc2_mu, inter_enc2_ln_var)
        
        #Decoder2 (300)
        inter_dec2 = tf.matmul(z, self.w_dec2) + self.b_dec2
        h_dec2 = tf.nn.tanh(inter_dec2)
        
        #Decoder1 (784)
        inter_dec1 = tf.matmul(h_dec2, self.w_dec1) + self.b_dec1
        h_dec1 = tf.nn.tanh(inter_dec1)
        h_dec1 = tf.reshape(h_dec1, [-1, 28, 28, 1])
        
        return h_dec1, inter_enc2_mu, inter_enc2_ln_var, z

def sample_z(mu, ln_var):
    epsilon = tf.random_normal(tf.shape(mu), 0.1, dtype=tf.float32)
    z = mu + epsilon * ln_var    
    return z
    
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


    
