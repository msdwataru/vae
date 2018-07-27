# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from BatchNormalization import BatchNormalization

from IPython import embed

class VAE:
    def __init__(self, ch_list=[3,32,32,16,8], image_size= 28, k_h=5, k_w=5, stddev=0.1):
        self.ch_list = ch_list
        self.image_size = image_size
        #define learnable parameter
        with tf.variable_scope("ae"):
            #encoder
            self.w_conv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[0], self.ch_list[1]], stddev= stddev))
            self.w_conv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev= stddev))
            
            self.w_enc3 = tf.Variable(tf.truncated_normal([4 * 4 * self.ch_list[2], self.ch_list[3]], stddev=stddev))
            self.b_enc3 = tf.Variable(tf.zeros([self.ch_list[3]]))
            self.w_enc4_mu = tf.Variable(tf.truncated_normal([self.ch_list[3], self.ch_list[4]], stddev=stddev))
            self.b_enc4_mu = tf.Variable(tf.zeros([self.ch_list[4]]))
            self.w_enc4_ln_var = tf.Variable(tf.truncated_normal([self.ch_list[3], self.ch_list[4]], stddev=stddev))
            self.b_enc4_ln_var = tf.Variable(tf.zeros([self.ch_list[4]]))
            
            #decoder
            self.w_dec4 = tf.Variable(tf.truncated_normal([self.ch_list[4], self.ch_list[3]],stddev=stddev))
            self.b_dec4 = tf.Variable(tf.zeros([self.ch_list[3]]))
            self.w_dec3 = tf.Variable(tf.truncated_normal([self.ch_list[3], 4 * 4 * self.ch_list[2]],stddev=stddev))
            self.b_dec3 = tf.Variable(tf.zeros([4 * 4 * self.ch_list[2]]))
            self.w_deconv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev=stddev))
            self.w_deconv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[0], self.ch_list[1]], stddev=stddev))

    def __call__(self, x, batch_size, train=True):
        #Conv1 (16 * 16 * 32) 
        h_conv1 = conv2d(x, self.w_conv1, train=train)

        #Conv2 (4 * 4 * 32) 
        h_conv2 = conv2d(h_conv1, self.w_conv2, train=train)

        # Full connection3 (100)
        h_conv2 = tf.reshape(h_conv2, [-1, 4 * 4 * 32])
        inter_enc3 = tf.matmul(h_conv2, self.w_enc3) + self.b_enc3
        h_enc3 = tf.nn.softplus(inter_enc3)

        #Full connection4 mu (10)
        inter_enc4_mu = tf.matmul(h_enc3, self.w_enc4_mu) + self.b_enc4_mu
        #h_enc2_mu = tf.nn.tanh(inter_enc2)

        #Full connection ln_var (10)
        inter_enc4_ln_var = tf.matmul(h_enc3, self.w_enc4_ln_var) + self.b_enc4_ln_var
        sigma = tf.exp(inter_enc4_ln_var)
        #h_enc2 = tf.nn.tanh(inter_enc2)

        #Sample z
        z = sample_z(inter_enc4_mu, sigma)
        
        #Decoder4 (100)
        inter_dec4 = tf.matmul(z, self.w_dec4) + self.b_dec4
        h_dec4 = tf.nn.softplus(inter_dec4)

        #Decoder3 (4 * 4 * 32)
        inter_dec3 = tf.matmul(h_dec4, self.w_dec3) + self.b_dec3
        h_dec3 = tf.nn.softplus(inter_dec3)
        h_dec3 = tf.reshape(h_dec3, [-1, 4, 4, self.ch_list[2]])

        #Deconv2(16 * 16 * 32)
        h_deconv2 = deconv2d(h_dec3, self.w_deconv2, [batch_size, 16, 16, self.ch_list[1]],train=train)

        #Deconv1(64 * 64 * 3)
        h_deconv1 = deconv2d(h_deconv2, self.w_deconv1, [batch_size, self.image_size, self.image_size, self.ch_list[0]],train=train, activation=tf.nn.sigmoid)

        return h_deconv1, inter_enc4_mu, sigma, z

def sample_z(mu, sigma):
    epsilon = tf.random_normal(tf.shape(mu), stddev=1.0, dtype=tf.float32)
    z = mu + epsilon * sigma
    return z
    
def conv2d(x, weight, batch_norm=None, train=True, activation=tf.nn.softplus):
    h_conv = tf.nn.conv2d(x, weight, strides=[1, 4, 4, 1], padding="SAME")
    if batch_norm != None:
        h_conv = batch_norm(h_conv, train=train)
    h_conv = activation(h_conv)
    return h_conv

def deconv2d(x, weight, output_shape, batch_norm=None, train=True, activation=tf.nn.softplus):
    h_deconv = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=[1, 4, 4, 1], padding="SAME")
    if batch_norm != None:
        h_deconv = batch_norm(h_deconv, train=train)
    h_deconv = activation(h_deconv)
    return h_deconv
