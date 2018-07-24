# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from BatchNormalization import BatchNormalization

from IPython import embed

class VAE:
    def __init__(self, ch_list=[3,32,32,16,8], image_size= 28, k_h=5, k_w=5, stddev=0.01):
        self.ch_list = ch_list
        self.image_size = image_size
        #define learnable parameter
        with tf.variable_scope("ae"):
            #encoder
            self.w_enc1 = tf.Variable(tf.truncated_normal([self.image_size * self.image_size * self.ch_list[0], self.ch_list[1]], stddev=stddev))
            self.b_enc1 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_enc2 = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[1]], stddev=stddev))
            self.b_enc2 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_enc3_mu = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.b_enc3_mu = tf.Variable(tf.zeros([self.ch_list[2]]))
            self.w_enc3_ln_var = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.b_enc3_ln_var = tf.Variable(tf.zeros([self.ch_list[2]]))
            
            #decoder
            self.w_dec3 = tf.Variable(tf.truncated_normal([self.ch_list[2], self.ch_list[1]],stddev=stddev))
            self.b_dec3 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_dec2 = tf.Variable(tf.truncated_normal([self.ch_list[1], self.ch_list[1]],stddev=stddev))
            self.b_dec2 = tf.Variable(tf.zeros([self.ch_list[1]]))
            self.w_dec1 = tf.Variable(tf.truncated_normal([self.ch_list[1], self.image_size * self.image_size * self.ch_list[0]], stddev=stddev))
            self.b_dec1 = tf.Variable(tf.zeros([self.image_size * self.image_size * self.ch_list[0]]))

    def __call__(self, x, batch_size, train=True):

        #Encoder1 (500)
        inputs = tf.reshape(x, [-1, self.ch_list[0] * self.image_size * self.image_size])
        inter_enc1 = tf.matmul(inputs, self.w_enc1) + self.b_enc1
        h_enc1 = tf.nn.softplus(inter_enc1)

        #Encoder2 (500)
        inter_enc2 = tf.matmul(h_enc1, self.w_enc2) + self.b_enc2
        h_enc2 = tf.nn.softplus(inter_enc2)

        #Encoder3 mu (20)
        inter_enc3_mu = tf.matmul(h_enc2, self.w_enc3_mu) + self.b_enc3_mu
        #h_enc2_mu = tf.nn.tanh(inter_enc2)

        #Encoder3 ln_var (20)
        inter_enc3_ln_var = tf.matmul(h_enc2, self.w_enc3_ln_var) + self.b_enc3_ln_var
        sigma = tf.exp(inter_enc3_ln_var)
        #h_enc2 = tf.nn.tanh(inter_enc2)

        #Sample z
        z = sample_z(inter_enc3_mu, sigma)
        
        #Decoder3 (500)
        inter_dec3 = tf.matmul(z, self.w_dec3) + self.b_dec3
        h_dec3 = tf.nn.softplus(inter_dec3)

        #Decoder2 (500)
        inter_dec2 = tf.matmul(h_dec3, self.w_dec2) + self.b_dec2
        h_dec2 = tf.nn.softplus(inter_dec2)
        
        #Decoder1 (784)
        inter_dec1 = tf.matmul(h_dec2, self.w_dec1) + self.b_dec1
        h_dec1 = tf.nn.sigmoid(inter_dec1)
        h_dec1 = tf.reshape(h_dec1, [-1, self.image_size, self.image_size, self.ch_list[0]])
        return h_dec1, inter_enc3_mu, sigma, z

def sample_z(mu, sigma):
    epsilon = tf.random_normal(tf.shape(mu), stddev=1.0, dtype=tf.float32)
    z = mu + epsilon * sigma
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
