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

WORK_DIRECTORY = '../dll/mnist/'
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10
SEED = None  # Set to None for random seed.
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 600  # Number of steps between evaluations.

FLAGS = None

def data_type():
    return tf.float32

def maybe_download(filename):
  filepath = os.path.join(WORK_DIRECTORY, filename)
  return filepath

def extract_data(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * 1)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def main(_):
  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)

  num_epochs = 50

  train_size = train_labels.shape[0]

  train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

  conv1_weights = tf.Variable( tf.truncated_normal([5, 5, 1, 8], stddev=0.1,dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([8], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 8, 8], stddev=0.1, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[8], dtype=data_type()))

  fc1_weights = tf.Variable(tf.truncated_normal([8 * 4 * 4, 150], stddev=0.1, dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[150], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([150, 10], stddev=0.1, dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[10], dtype=data_type()))

  def model(data, train=False):
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
    sig = tf.nn.sigmoid(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(sig, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
    sig = tf.nn.sigmoid(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(sig, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    flat = tf.reshape(pool, [BATCH_SIZE, 8 * 4 * 4])
    hidden = tf.nn.sigmoid(tf.matmul(flat, fc1_weights) + fc1_biases)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = train_labels_node))

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss)

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
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
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
              (float(step) * BATCH_SIZE / train_size, l, error_rate(eval_in_batches(train_data, sess), train_labels)), flush=True)
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error, flush=True)

tf.app.run(main=main, argv=[sys.argv[0]])
