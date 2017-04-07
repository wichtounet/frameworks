# From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

cifar10_dataset_folder_path = 'cifar-10-batches-py'
BATCH_SIZE = 100
EVAL_FREQUENCY = 500  # Number of steps between evaluations.
num_epochs = 50

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

def main(_):
  if not isfile('cifar-10-python.tar.gz'):
      print("Download data...")
      urlretrieve(
          'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
          'cifar-10-python.tar.gz'
          )

  if not isdir(cifar10_dataset_folder_path):
      print("Extract data...")
      with tarfile.open('cifar-10-python.tar.gz') as tar:
          tar.extractall()
          tar.close()

  print("Data is ready...")

  for batch_i in range(1, 6):
      with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_i), mode='rb') as file:
          batch = pickle.load(file, encoding='latin1')

      features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
      labels = batch['labels']

      labels = numpy.array(labels)

      if batch_i == 1:
          train_data = features
          train_labels = labels
      else:
          train_data = numpy.concatenate([train_data, features])
          train_labels = numpy.concatenate([train_labels, labels])

  with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')

  test_data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
  test_labels = batch['labels']
  test_labels = numpy.array(test_labels)

  train_data = train_data.astype(numpy.float32)
  test_data = test_data.astype(numpy.float32)

  train_data = train_data * (1.0 / 255.0)
  test_data = test_data * (1.0 / 255.0)

  train_size = train_labels.shape[0]

  train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, 32, 32, 3))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(data_type(), shape=(BATCH_SIZE, 32, 32, 3))

  # First layer weights
  conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 3, 12], stddev=0.1, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([12], dtype=data_type()))

  # Second layer weights
  conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 12, 24], stddev=0.1, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[24], dtype=data_type()))

  # Fully connected weights
  fc1_weights = tf.Variable(tf.truncated_normal([24 * 6 * 6, 64], stddev=0.1, dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1, dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[10], dtype=data_type()))

  def model(data, train=False):
    # Conv 1
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv 2
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully Connected
    reshape = tf.reshape(pool, [BATCH_SIZE, 24 * 6 * 6])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = train_labels_node))

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, 10), dtype=numpy.float32)
    for begin in xrange(0, size, BATCH_SIZE):
      end = begin + BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run(session = sess)
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, predictions = sess.run([loss, train_prediction], feed_dict=feed_dict)
        print('Epoch %.2f Minibatch loss: %.3f Train error: %.1f%%' %
              (float(step) * BATCH_SIZE / train_size, l, error_rate(eval_in_batches(train_data, sess), train_labels)))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)

tf.app.run(main=main, argv=[sys.argv[0]])
