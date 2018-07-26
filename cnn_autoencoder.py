# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from BatchNormalization import BatchNormalization

from IPython import embed

class CNNAE:
    def __init__(self, k_h, k_w, ch_list=[3,32,32,16,8], stddev=0.01):
        self.ch_list = ch_list
        #define learnable parameter
        with tf.variable_scope("cnn"):
            #conv
            self.w_conv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[0], self.ch_list[1]], stddev=stddev))
            
            self.w_conv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev=stddev))
            
            self.w_conv3 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[2], self.ch_list[3]], stddev=stddev))
            
            self.w_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * self.ch_list[3], 5]))
            self.b_fc1 = tf.Variable(tf.zeros([5]))

            #deconv
            self.w_fc2 = tf.Variable(tf.truncated_normal([5, 128]))
            self.b_fc2 = tf.Variable(tf.zeros([128]))

            self.w_deconv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[2], self.ch_list[3]], stddev=stddev))

            self.w_deconv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev=stddev))
            
            self.w_deconv3 = tf.Variable(tf.truncated_normal([k_h, k_w,self.ch_list[0], self.ch_list[1]], stddev=stddev))

    def __call__(self, x, batch_size, train=True):
        #Conv1(14*14*16)
        h_conv1 = conv2d(x, self.w_conv1, train=train)
        
        #Conv2(7*7*16)
        h_conv2 = conv2d(h_conv1, self.w_conv2, train=train)

        #Conv3(4*4*8)
        h_conv3 = conv2d(h_conv2, self.w_conv3, train=train)

        #Full connection1(10)
        h_conv3 = tf.reshape(h_conv3, [-1, 4 * 4 * self.ch_list[3]])
        h_fc1 = tf.matmul(h_conv3, self.w_fc1) + self.b_fc1
        h_fc1 = tf.nn.tanh(h_fc1)

        #Full connection2(128)
        h_fc2 = tf.matmul(h_fc1, self.w_fc2) + self.b_fc2
        h_fc2 = tf.nn.tanh(h_fc2)
        h_fc2 = tf.reshape(h_fc2, [-1, 4, 4, self.ch_list[3]])

        #Deconv1(7*7*16)
        h_deconv1 = deconv2d(h_fc2, self.w_deconv1, [batch_size, 7, 7, self.ch_list[2]], train=train)
        
        #Deconv2(14*14*16)
        h_deconv2 = deconv2d(h_deconv1, self.w_deconv2, [batch_size, 14, 14, self.ch_list[1]], train=train)
        
        #Deconv3(28*28*1)
        h_deconv3 = deconv2d(h_deconv2, self.w_deconv3, [batch_size, 28, 28, self.ch_list[0]], train=train)

        return h_deconv3, h_fc1
        
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


    
