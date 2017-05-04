from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import argparse
import gzip
import os
import sys
import time

import os
import math
import numpy
from PIL import Image

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

batch_size = 128
batches = 10009
EVAL_FREQUENCY = 10009  # Number of steps between evaluations.
num_epochs = 10
num_classes = 1000

FLAGS = None

from urllib.request import urlretrieve
from os.path import isfile, isdir
import tarfile
import pickle

def data_type():
    return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def get_batch():
    index = 1

    global current_index
    global training_images
    global training_labels

    B = numpy.zeros(shape=(batch_size, 256, 256, 3))
    L = numpy.zeros(shape=(batch_size))

    while index < batch_size:
        img = load_img(training_images[current_index])
        B[index] = img_to_array(img)
        B[index] /= 255

        L[index] = training_labels[current_index]

        index = index + 1
        current_index = current_index + 1

    return B, keras.utils.to_categorical(L, num_classes)

def main(_):
    global current_index
    global training_images
    global training_labels

    label_counter = 0

    training_images = []
    training_labels = []

    for subdir, dirs, files in os.walk('/data/datasets/imagenet_resized/train/'):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(label_counter)

            label_counter = label_counter + 1

    nice_n = math.floor(len(training_images) / batch_size) * batch_size

    print(nice_n)
    print(len(training_images))
    print(len(training_labels))

    import random

    perm = list(range(len(training_images)))
    random.shuffle(perm)
    training_images = [training_images[index] for index in perm]
    training_labels = [training_labels[index] for index in perm]

    print("Data is ready...")

    train_data_node = tf.placeholder(data_type(), shape=(batch_size, 256, 256, 3))
    train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,1000))

    # Convolutional weights
    conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([16], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 16, 16], stddev=0.1, dtype=data_type()))
    conv2_biases = tf.Variable(tf.zeros([16], dtype=data_type()))
    conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=data_type()))
    conv3_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1, dtype=data_type()))
    conv4_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1, dtype=data_type()))
    conv5_biases = tf.Variable(tf.zeros([32], dtype=data_type()))

    # Fully connected weights
    fc1_weights = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.1, dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[2048], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([2048, 1000], stddev=0.1, dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[1000], dtype=data_type()))

    def model(data):
        # Conv 1
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Conv 2
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Conv 3
        conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Conv 4
        conv = tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Conv 5
        conv = tf.nn.conv2d(pool, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully Connected
        reshape = tf.reshape(pool, [batch_size, 2048])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = train_labels_node))

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

    acc_pred = tf.equal(tf.argmax(logits,1), tf.argmax(train_labels_node,1))
    accuracy = tf.reduce_mean(tf.cast(acc_pred, tf.float32))

    # Predictions for the current training minibatch.
    # train_prediction = tf.nn.softmax(logits)

    # Create a local session to run the training.
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run(session = sess)
        print('Initialized!')

        for epoch in range(0, num_epochs):
            current_index = 0

            while current_index + batch_size < len(training_images):
                b, l = get_batch()

                feed_dict = {train_data_node: b, train_labels_node: l}

                # Run the optimizer to update weights.
                _, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)

                print('batch {}/{} loss: {} accuracy: {}'.format(int(current_index / batch_size), int(nice_n / batch_size), batch_loss, batch_accuracy))
                sys.stdout.flush()

            print('epoch {}/{}'.format(epoch, num_epoch))

        # Finally print the result!

        current_index = 0
        acc = 0.0

        while current_index + batch_size < len(training_images):
            b, l = get_batch()

            feed_dict = {train_data_node: b, train_labels_node: l}
            [batch_accuracy] = sess.run([accuracy], feed_dict=feed_dict)

            acc += batch_accuracy

        acc /= 10009

        print('Test accuracy: %.1f%%' % acc)

tf.app.run(main=main, argv=[sys.argv[0]])
