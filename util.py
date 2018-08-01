#! -*- coding:utf-8 -*-

import glob
import numpy as np
import cv2
import random
import time
import os
import sys
from IPython import embed

def read_images(data_dir, use_labels=False):
    files = os.listdir(data_dir)
    files_dir = [f for f in files if os.path.isdir(os.path.join(data_dir, f))]
    files_dir = [data_dir + f for f in files_dir]
    dir_num = len(files_dir)
    images = []
    for file_dir in files_dir:
        #for image in glob.glob(file_dir + "/*.jpg")[1300:1305]:
        for image in glob.glob(file_dir + "/*.jpg"):
            images.append(image)
    #images = glob.glob(data_dir + "*")
    #random.shuffle(images)
    #images.sort()
    sample_image = cv2.imread(images[0])
    data = np.empty([len(images), sample_image.shape[0], sample_image.shape[1], sample_image.shape[2]])
    labels = []
    for i, image in enumerate(images):
        #print("reading a file: {0}".format(file))
        rgb_image = cv2.imread(image)
        #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).reshape(28,28,1)
        #data.append(rgb_image)
        #if rgb_image.shape != (256, 256, 3):
            #print image
        data[i, :, :, :] = rgb_image
        if use_labels:
            file_name = image.split("/")[-1]
            labels.append(int(re.findall(r"\d+", file_name)[0][:3]))
        process_rate = 100. * (i + 1) / len(images)
        sys.stdout.write("\r loading images [{0:<20}] {1:3d}%"
                         .format(int(process_rate//5) * "=",
                                 int(process_rate)))
    sys.stdout.write("\n")
    data_num = len(data)
    data_shape = data[0].shape
    
    print("data num = {}".format(data_num))
    print("data shape = {}".format(data_shape))
    if use_labels:
        return data, data_num, data_shape, labels
    else:
        return data, data_num, data_shape, dir_num

def read_goal_images(data_dir):
    dir_path = glob.glob(data_dir + "goal*")
    images = []
    for dp in dir_path:
        for image in glob.glob(dp + "/goal*.png"):
            images.append(image)
    random.shuffle(images)
    #images.sort()
    sample_image = cv2.imread(images[0])
    data = np.empty([len(images), sample_image.shape[0], sample_image.shape[1], sample_image.shape[2]])
    labels = []
    for i, image in enumerate(images):
        #print("reading a file: {0}".format(file))
        rgb_image = cv2.imread(image)
        #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).reshape(28,28,1)
        #data.append(rgb_image)
        data[i, :, :, :] = rgb_image
        file_name = image.split("/")[-1]
        labels.append(int(re.findall(r"\d+", file_name)[0][:3]))        

    data_num = len(data)
    data_shape = data[0].shape
        
    print("data num = {}".format(data_num))
    print("data shape = {}".format(data_shape))
    return data, data_num, data_shape, labels

def normalize(x, min_in=0, max_in=255, min_out=-1, max_out=1, scale=0.8):
    x = (x - min_in) / float(max_in - min_in)
    x = scale * x * (max_out - min_out) + min_out
    return x

def denormalize(x, min_in=0, max_in=255, min_out=-1, max_out=1, scale=0.8):
    x = (x / scale - min_out) / (max_out - min_out) * (max_in - min_in)
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
    def __call__(self, epoch, loss, latent_loss=None):
        current_time = time.time()
        self.total_time = current_time - self.start_time
        if latent_loss != None:
            print("epoch: {} time: {} loss: {} latent loss: {}".format(epoch, self.total_time, loss, latent_loss))
        else:
            print("epoch: {} time: {} loss: {}".format(epoch, self.total_time, loss))
        self.error_arr = np.r_[self.error_arr, loss]
            
