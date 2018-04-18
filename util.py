#! -*- coding:utf-8 -*-

import glob
import numpy as np
import cv2
import random
import time

import sys
from IPython import embed

def read_image(data_dir):
    images = glob.glob(data_dir + "*png")
    #images = glob.glob(data_dir + "*")
    random.shuffle(images)
    #images.sort()
    sample_image = cv2.imread(images[0])
    data = np.empty([len(images), sample_image.shape[0], sample_image.shape[1], sample_image.shape[2]])
    for i, image in enumerate(images):
        #print("reading a file: {0}".format(file))
        rgb_image = cv2.imread(image)
        #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).reshape(28,28,1)
        #data.append(rgb_image)
        data[i, :, :, :] = rgb_image
    data_num = len(data)
    data_shape = data[0].shape
    
    print("data num = {}".format(data_num))
    print("data shape = {}".format(data_shape))
    return data, data_num, data_shape

def normalize(x, min_in=0, max_in=255, min_out=-1, max_out=1):
    x = (x - min_in) / float(max_in - min_in)
    x = x * (max_out - min_out) + min_out
    return x

def denormalize(x, min_in=0, max_in=255, min_out=-1, max_out=1):
    x = (x - min_out) / (max_out - min_out) * (max_in - min_in)
    return x

def add_noise(x, sigma=0.1):
    gaussian_noise = np.random.normal(0, sigma, size=(x.shape))
    x += gaussian_noise
    return x
    
class Logger():
    def __init__(self):
        self.total_time = 0
        self.start_time = time.time()
        self.error_arr = np.zeros((0))
    def __call__(self, epoch, loss):
        current_time = time.time()
        self.total_time = current_time - self.start_time
        print("epoch: {} time: {} loss: {}".format(epoch + 1, self.total_time, loss))
        self.error_arr = np.r_[self.error_arr, loss]
            
