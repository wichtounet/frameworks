import tensorflow as tf
import numpy as np
import input_data

def sample_prob(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    # return tf.select((tf.random_uniform(tf.shape(probs), 0, 1) - probs) > 0.5, tf.ones(tf.shape(probs)), tf.zeros(tf.shape(probs)))

learning_rate = 0.01
momentum = 0.9
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", validation_size=0, one_hot=False)
trX = mnist.train.images

trX = np.reshape(trX, (60000, 28, 28, 1))

X = tf.placeholder("float", [100, 28, 28, 1], name = "X")

rbm_w = tf.placeholder("float", [5, 5, 1, 8], name = "rbm_w")
rbm_vb = tf.placeholder("float", [1], name = "rbm_vb")
rbm_hb = tf.placeholder("float", [8], name = "rbm_hb")

rbm_w_inc = tf.placeholder("float", [5, 5, 1, 8], name = "rbm_w_inc")
rbm_vb_inc = tf.placeholder("float", [1], name = "rbm_vb_inc")
rbm_hb_inc = tf.placeholder("float", [8], name = "rbm_hb_inc")

h0_x = tf.nn.conv2d(X, rbm_w, strides=[1, 1, 1, 1], padding='VALID')
h0_a = tf.nn.sigmoid(tf.nn.bias_add(h0_x, rbm_hb))
h0 = sample_prob(h0_a)

v1_x = tf.nn.conv2d_transpose(h0, rbm_w, output_shape=X.get_shape(), strides=[1, 1, 1, 1], padding='VALID')
v1_a = tf.nn.sigmoid(tf.nn.bias_add(v1_x, rbm_vb))
v1 = sample_prob(v1_a)

h1_x = tf.nn.conv2d(v1, rbm_w, strides=[1, 1, 1, 1], padding='VALID')
h1_a = tf.nn.sigmoid(tf.nn.bias_add(h1_x, rbm_hb))
h1 = sample_prob(h1_a)

w_positive_grad = tf.nn.conv2d_backprop_filter(X, filter_sizes=rbm_w.get_shape(), out_backprop=h0_a, strides=[1, 1, 1, 1], padding='VALID')
w_negative_grad = tf.nn.conv2d_backprop_filter(v1_a, filter_sizes=rbm_w.get_shape(), out_backprop=h1_a, strides=[1, 1, 1, 1], padding='VALID')

grad_w = (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
grad_vb = tf.reduce_mean(X - v1_a)
grad_hb = tf.reduce_mean(h0 - h1_a, (0, 1, 2))

update_w_inc = momentum * rbm_w_inc + (learning_rate / batchsize) * grad_w
update_vb_inc = momentum * rbm_vb_inc + (learning_rate / batchsize) * grad_vb
update_hb_inc = momentum * rbm_hb_inc + (learning_rate / batchsize) * grad_hb

update_w = rbm_w + update_w_inc
update_vb = rbm_vb + update_vb_inc
update_hb = rbm_hb + update_hb_inc

err = X - v1_a
err_sum = tf.reduce_mean(err * err)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

o_w = np.zeros([5, 5, 1, 8], np.float32)
o_vb = np.zeros([1], np.float32)
o_hb = np.zeros([8], np.float32)

o_w_inc = np.zeros([5, 5, 1, 8], np.float32)
o_vb_inc = np.zeros([1], np.float32)
o_hb_inc = np.zeros([8], np.float32)

error = 0.0
for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    error += sess.run(err_sum, feed_dict={X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
print(error / 600.0)

for e in range(0, 50):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        o_w_inc, o_vb_inc, o_hb_inc, o_w, o_vb, o_hb = sess.run( \
                [update_w_inc, update_vb_inc, update_hb_inc, update_w, update_vb, update_hb], \
                feed_dict={X: batch, rbm_w_inc: o_w_inc, rbm_vb_inc: o_vb_inc, rbm_hb_inc: o_hb_inc, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})

    error = 0.0
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        error += sess.run(err_sum, feed_dict={X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    print(error / 600.0)
