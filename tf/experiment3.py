import tensorflow as tf
import numpy as np
import input_data

def sample_prob(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    # return tf.select((tf.random_uniform(tf.shape(probs), 0, 1) - probs) > 0.5, tf.ones(tf.shape(probs)), tf.zeros(tf.shape(probs)))

learning_rate = 0.1
momentum = 0.9
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])

rbm_w_inc = tf.placeholder("float", [784, 500])
rbm_vb_inc = tf.placeholder("float", [784])
rbm_hb_inc = tf.placeholder("float", [500])

h0_a = tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb)
h0 = sample_prob(h0_a)

v1_a = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb)
v1 = sample_prob(v1_a)

h1_a = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)
# h1 = sample_prob(h1_a)

w_positive_grad = tf.matmul(tf.transpose(X), h0_a)
w_negative_grad = tf.matmul(tf.transpose(v1_a), h1_a)

grad_w = (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
grad_vb = tf.reduce_mean(X - v1_a, 0)
grad_hb = tf.reduce_mean(h0 - h1_a, 0)

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

o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)

o_w_inc = np.zeros([784, 500], np.float32)
o_vb_inc = np.zeros([784], np.float32)
o_hb_inc = np.zeros([500], np.float32)

print(sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb}))

for e in range(0, 50):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        o_w_inc, o_vb_inc, o_hb_inc, o_w, o_vb, o_hb = sess.run([update_w_inc, update_vb_inc, update_hb_inc, update_w, update_vb, update_hb], feed_dict={X: batch, rbm_w_inc: o_w_inc, rbm_vb_inc: o_vb_inc, rbm_hb_inc: o_hb_inc, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})

    print(sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb}))
