# -*- coding:utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import time

from autoencoder import Autoencoder
from cnn_autoencoder import CNNAE
from vae_with_conv_layer import VAE
from util import *

from IPython import embed

#option
flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epoch")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("k_h", 3, "Kernel height")
flags.DEFINE_integer("k_w", 3, "Kernel width")
flags.DEFINE_integer("seed", 20180417, "Random seed")
flags.DEFINE_integer("li", 100, "Log interval")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_boolean("is_minibatch", False, "Apply minibatch training or batch training")
flags.DEFINE_string("data_dir", "./image/env_images/", "Directory of training data")
flags.DEFINE_string("test_data_dir", "./data/baxter_image/test/", "Directory of test data")
flags.DEFINE_string("save_dir", "./result", "Directory which results are saved")

FLAGS = flags.FLAGS

def make_placeholder(data_shape, is_training=True):
    input_ph = tf.placeholder(tf.float32, shape=[None, data_shape[0], data_shape[1], data_shape[2]], name="input")
    target_ph = tf.placeholder(tf.float32, shape=[None, data_shape[0], data_shape[1], data_shape[2]], name="target")
    return input_ph, target_ph

def _loss(outputs, targets):
    with tf.name_scope("loss") as loss:
        loss = tf.reduce_mean(tf.square(outputs - targets))
        return loss

def _loss_with_KL_divergence(outputs, targets, mu, sigma):
    with tf.name_scope("loss") as loss:
        #reconstruction_loss = 0.5 * tf.reduce_mean(tf.square(outputs - targets))
        reconstruction_loss = -tf.reduce_sum(targets * tf.log(1e-7 + outputs) + (1 - targets) * tf.log(1e-7 + 1 - outputs), [1, 2, 3])
        #reconstruction_loss = -tf.reduce_mean(targets * tf.log(1e-10 + outputs))
        #reconstruction_loss = 0.5 * tf.reduce_mean(tf.square(outputs - targets))
        latent_loss = 0.5 * tf.reduce_sum(sigma + tf.square(mu) - tf.log(1e-7 + sigma) - 1, 1)
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
    #data, data_num, data_shape, labels = read_images(FLAGS.data_dir, use_labels=True)
    data, data_num, data_shape = read_images(FLAGS.data_dir)
    #test_data,test_data_num, test_data_shape = read_image(FLAGS.test_data_dir)
    input_placeholder, target_placeholder = make_placeholder(data_shape)
    channel_list = [data_shape[2], 32, 32, 100,  10]
    #ae = Autoencoder(ch_list=channel_list)
    vae = VAE(ch_list=channel_list, image_size=data_shape[0])
    outputs, mu, sigma, latent_variable = vae(input_placeholder, FLAGS.batch_size, train=True) if FLAGS.is_minibatch else vae(input_placeholder, data_num, train=True)
    loss, rec_loss, la_loss = _loss_with_KL_divergence(outputs, target_placeholder, mu, sigma)
    train_op = _train(loss)
    logger = Logger()
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_xs = normalize(data, min_out=0., max_out=1., scale=1.)
        #batch_ts = np.empty(data.shape)
        #for i in range(len(data)):
        #    batch_ts[i] = batch_xs[i / 10]
        #start training
        for epoch in range(1, FLAGS.epoch + 1):
            if FLAGS.is_minibatch:
                avg_error = 0
                avg_lat_error = 0
                total_batch = int(data_num / FLAGS.batch_size)
                for i in range(total_batch):
                    mini_batch_xs = batch_xs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :]
                    feed_dict = {input_placeholder: mini_batch_xs, target_placeholder: mini_batch_xs}

                    result = sess.run([loss, rec_loss, la_loss, sigma, train_op], feed_dict=feed_dict)
                    avg_error += result[0] / data_num * FLAGS.batch_size
                    avg_lat_error += result[2] / data_num * FLAGS.batch_size
                logger(epoch, avg_error, latent_loss=avg_lat_error)

            else:
                #noised_batch_xs = add_noise(batch_xs)
                feed_dict = {input_placeholder: batch_xs, target_placeholder: batch_xs}
                result = sess.run([loss,rec_loss, la_loss, sigma, train_op], feed_dict=feed_dict)
                if epoch % 10 == 0:
                    #test_batch_xs = normalize(test_data[:FLAGS.batch_size])
                    #test_loss = sess.run(loss, feed_dict={input_placeholder: test_batch_xs, target_placeholder: test_batch_xs})
                    #logger(i, result[0], test_loss)
                    logger(epoch, result[0], latent_loss=result[2])
                
            if epoch % FLAGS.li == 0:
                saver.save(sess, FLAGS.save_dir + "/model", global_step = epoch)

        np.savetxt(FLAGS.save_dir + "/error.log", logger.error_arr, fmt="%0.6f")

        if FLAGS.is_minibatch:
            reconstructed_images, mu_log = sess.run([outputs, mu], feed_dict={input_placeholder: batch_xs[:FLAGS.batch_size], target_placeholder: batch_xs[:FLAGS.batch_size]})
        else:
            reconstructed_images, mu_log = sess.run([outputs, mu], feed_dict={input_placeholder: batch_xs, target_placeholder: batch_xs})         
            #mu_and_labels = np.c_[mu_log, labels]
        #np.savetxt(FLAGS.save_dir + "/latent_variables.log", mu_and_labels)
        embed()
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    batch_xs = denormalize(batch_xs, min_out=0., max_out=1., scale=1.)
    reconstructed_images = denormalize(reconstructed_images, min_out=0., max_out=1., scale=1.)
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        #original = batch_xs[i].reshape(28, 28)
        plt.imshow(cv2.cvtColor(batch_xs[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        #plt.imshow(original, cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        #rec = reconstructed_images[i].reshape(28,28)
        plt.imshow(cv2.cvtColor(reconstructed_images[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        #plt.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(FLAGS.save_dir + '/result.png')

    plt.show()
        
if __name__ == "__main__":
    tf.app.run()
