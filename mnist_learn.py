# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import time
import sys

from autoencoder import Autoencoder
from cnn_autoencoder import CNNAE
from vae import VAE
from util import *

from IPython import embed

#option
flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Number of epoch")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("k_h", 3, "Kernel height")
flags.DEFINE_integer("k_w", 3, "Kernel width")
flags.DEFINE_integer("seed", 20180417, "Random seed")
flags.DEFINE_integer("li", 10, "Log interval")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_string("data_dir", "./data/baxter_image/target/20171129/", "Directory of training data")
flags.DEFINE_string("test_data_dir", "./data/baxter_image/test/", "Directory of test data")
flags.DEFINE_string("save_dir", "./result", "Directory which results are saved")

FLAGS = flags.FLAGS

def make_placeholder(data_shape, is_training=True):
    input_ph = tf.placeholder(tf.float32, shape=[None, data_shape[0], data_shape[1], data_shape[2]], name="input")
    target_ph = tf.placeholder(tf.float32, shape=[None, data_shape[0], data_shape[1], data_shape[2]], name="target")
    return input_ph, target_ph

def _loss(outputs, targets):
    with tf.name_scope("loss") as loss:
        loss = tf.reduce_sum(tf.square(outputs - targets))
        return loss

def _loss_with_KL_divergence(outputs, targets, mu, sigma):
    with tf.name_scope("loss") as loss:
        #reconstruction_loss = 0.5 * tf.reduce_mean(tf.square(outputs - targets))
        reconstruction_loss = -tf.reduce_sum(targets * tf.log(1e-10 + outputs) + (1 - targets) * tf.log(1e-10 + 1 - outputs), [1, 2])
        #reconstruction_loss = 0.5 * tf.reduce_mean(tf.square(outputs - targets))
        latent_loss = 0.5 * tf.reduce_sum(sigma + tf.square(mu) - tf.log(sigma) - 1, 1)
        #latent_loss = 0.5 * tf.reduce_sum(tf.square(sigma) + tf.square(mu) - tf.log(tf.square(sigma)) - 1)
        #latent_loss = 0.5 * tf.reduce_sum(sigma + tf.square(mu) - tf.log(sigma) - 1, [1])
        loss = tf.reduce_mean(reconstruction_loss + 1. * latent_loss)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        latent_loss = tf.reduce_mean(latent_loss)
        #loss = reconstruction_loss + 1.0 * latent_loss
        return loss, reconstruction_loss, latent_loss

    
def _train(loss):
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss)
    return train_op

def main(_):
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_samples = mnist.train.num_examples
    data, _ = mnist.train.next_batch(50)
    data_shape = [28, 28, 1]
    #data, data_num, data_shape = read_image(FLAGS.data_dir)
    #test_data,test_data_num, test_data_shape = read_image(FLAGS.test_data_dir)
    input_placeholder, target_placeholder = make_placeholder(data_shape)

    #channel_list = [1, 500, 20]
    channel_list = [1, 16, 16, 8]
    cnn = CNNAE(k_h=FLAGS.k_h, k_w=FLAGS.k_w, ch_list=channel_list)
    #ae = Autoencoder(ch_list=channel_list)
    #vae = VAE(ch_list=channel_list)
    #outputs, mu, sigma, latent_variable = vae(input_placeholder, FLAGS.batch_size, train=True)
    outputs, features = cnn(input_placeholder, FLAGS.batch_size)
    loss = _loss(outputs, target_placeholder)
    #loss, rec_loss, la_loss = _loss_with_KL_divergence(outputs, target_placeholder, mu, sigma)
    train_op = _train(loss)
    logger = Logger()
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #start training
        for epoch in range(1, FLAGS.epoch + 1):
            avg_error = 0
            avg_lat_error = 0
            total_batch = int(n_samples / FLAGS.batch_size)
            for i in range(total_batch):
                batch_xs, labels = mnist.train.next_batch(FLAGS.batch_size)
                batch_xs = np.reshape(batch_xs, [FLAGS.batch_size, 28, 28, 1])
                #noised_batch_xs = add_noise(batch_xs)
                #result = sess.run([loss,rec_loss, la_loss, train_op], feed_dict={input_placeholder: batch_xs, target_placeholder: batch_xs})
                result = sess.run([loss, train_op], feed_dict={input_placeholder: batch_xs, target_placeholder: batch_xs})
                avg_error += result[0] / n_samples * FLAGS.batch_size
                #avg_lat_error += result[2] / n_samples * FLAGS.batch_size
                process_rate = 100. * (i + 1) / total_batch
                sys.stdout.write("\rminibatch {0} [{1:<20}] {2:3d}%"
                                 .format(i, int(process_rate//5) * "=",
                                         int(process_rate)))
            sys.stdout.write("\n")
            #logger(epoch, avg_error, latent_loss=avg_lat_error)
            logger(epoch, avg_error)
                
            if epoch % FLAGS.li == 0:
                saver.save(sess, FLAGS.save_dir + "/model", global_step = i)

        np.savetxt(FLAGS.save_dir + "/error.log", logger.error_arr, fmt="%0.6f")

        batch_xs, labels = mnist.train.next_batch(100)
        batch_xs = np.reshape(batch_xs, [100, 28, 28, 1])
        reconstructed_images = sess.run(outputs, feed_dict={input_placeholder: batch_xs, target_placeholder: batch_xs})
        #reconstructed_images, mu_log = sess.run([outputs, mu], feed_dict={input_placeholder: batch_xs, target_placeholder: batch_xs})
        #mu_and_labels = np.c_[mu_log, labels]
        #np.savetxt(FLAGS.save_dir + "/latent_variables.log", mu_and_labels)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)

        original = batch_xs[i].reshape(28, 28)
        #plt.imshow(cv2.cvtColor(batch_xs[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.imshow(original, cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        rec = reconstructed_images[i].reshape(28,28)
        #plt.imshow(cv2.cvtColor(reconstructed_images[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(FLAGS.save_dir + '/result.png')

    plt.show()
        
if __name__ == "__main__":
    tf.app.run()
