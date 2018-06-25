#! -*- coding:utf-8 -*-
import glob
import numpy as np
import cv2
import re
import sys
from IPython import embed
import random

def read(net_conf, train_conf):
    data = []
    length = []
    filenames = glob.glob(train_conf.data_dir + "target*")
    filenames.sort()
    for file in filenames:
        print("reading a file: {0}".format(file))
        data.append(np.loadtxt(file, dtype=np.float32))
        length.append(data[-1].shape[0])

    n_dataset = len(data)
    batch_length = data[0].shape[0]
    if n_dataset > 1:
        for i in range(n_dataset - 1):
            if batch_length < data[i + 1].shape[0]:
                batch_length = data[i + 1].shape[0]

    batch_size = n_dataset
    batch = np.zeros((batch_size, batch_length, data[0].shape[1]))
    binaries = np.zeros((batch_size, batch_length, data[0].shape[1]))

    for b in range(batch_size):
        for i in range(data[b].shape[0]):
        #index = np.random.randint(0, n_dataset)
        #start = np.random.randint(0, data[index].shape[0]-batch_length)
            batch[b,i,:] = data[b][i,:]
            binaries[b,i,:] = 1
    print(batch.shape)
    print("data NUM is {}.".format(batch_size))
    print("max data LENGTH is {}.".format(batch_length))

    return batch, batch_size, batch_length, binaries

def read_for_generate(net_conf, train_conf):
    data = []
    length = []
    filenames = glob.glob(train_conf.data_dir + "test*")
    filenames.sort()
    for file in filenames:
        print("reading a file: {}".format(file))
        data.append(np.loadtxt(file, dtype=np.float32))
        length.append(data[-1].shape[0])

    min_length = min(length)
    n_dataset = len(data)

    batch = np.full((n_dataset, min_length, data[0].shape[1]), -1.0, dtype=np.float32)
    for b in range(n_dataset):
        batch[b,:,:] = data[b][:min_length]
    print(batch.shape)

    print("data NUM is {}.".format(n_dataset))
    print("data LENGTH is {}.".format(min_length))
    
    return batch, n_dataset, min_length


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
                                                                                                                       
