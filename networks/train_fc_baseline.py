from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import sys
import numpy as np
import tempfile

import tensorflow as tf
import matplotlib.pyplot as plt


def concatenate_training_data():
    logged_input_files = glob.glob("good_logs/logged_inputs*")
    print(logged_input_files)
    logged_inputs = np.concatenate([np.load(file) for file in logged_input_files], axis = 0)
    print("logged_inputs: ", logged_inputs.shape)

    logged_state_files = glob.glob("good_logs/logged_states*")
    logged_states = np.concatenate([np.load(file) for file in logged_state_files], axis = 0)
    print("logged_states: ", logged_states.shape)

    return logged_states, logged_inputs

logged_states, logged_inputs = concatenate_training_data()
num_samples    = logged_states.shape[0]
random_indices = np.random.permutation(num_samples)
train_indices  = random_indices[:(num_samples - 20000)]
valid_indices  = random_indices[(num_samples-20000):(num_samples-10000)]
test_indices   = random_indices[(num_samples-10000):]

# Data preprocess
# accel, brake, gear, steer
num_inputs = 1
input_idxs = [3]
states_idxs = [0] + list(range(54, 74))
num_states = len(states_idxs)

train_dataset = logged_states[train_indices, :]
train_dataset = train_dataset[:, states_idxs]

train_labels  = logged_inputs[train_indices, :]
train_labels  = train_labels[:, input_idxs].reshape((-1, num_inputs))

valid_dataset = logged_states[valid_indices, :]
valid_dataset = valid_dataset[:, states_idxs]

valid_labels  = logged_inputs[valid_indices, :]
valid_labels  = valid_labels[:, input_idxs].reshape((-1, num_inputs))

test_dataset  = logged_states[test_indices, :]
test_dataset  = test_dataset[:, states_idxs]

test_labels   = logged_inputs[test_indices, :]
test_labels   = test_labels[:, input_idxs].reshape((-1, num_inputs))

# num_states = train_dataset.shape[1]
batch_size = 32

def get_next_batch():
    batch_indices = np.random.randint(train_labels.shape[0], size = batch_size)
    batch_dataset = train_dataset[batch_indices, :]
    batch_labels = train_labels[batch_indices, :]
    return batch_dataset, batch_labels

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, num_states], name="x")
    y = tf.placeholder(tf.float32, [None, num_inputs], name="y")

    x_norm = tf.nn.l2_normalize(x, dim = 1, epsilon=1e-12, name = 'x_norm')
    # y_norm = tf.nn.l2_normalize(y, dim = 1, epsilon=1e-12, name = 'y_norm')

    fc1 = tf.nn.relu(tf.layers.dense(x_norm, 256))
    fc1 = tf.layers.batch_normalization(fc1)

    fc2 = tf.nn.relu(tf.layers.dense(fc1, 256))
    fc2 = tf.layers.batch_normalization(fc2)
    # fc2 = tf.nn.batch_normalization(fc2)

    predictions = tf.layers.dense(fc2, num_inputs, name="predictions")

    with tf.name_scope('loss'):
        mse_loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(mse_loss)

    losses, val_losses = [], []


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(25000):
            batch = get_next_batch()
            # print(batch[1])
            _, loss = sess.run([train_step, mse_loss], feed_dict={x: batch[0], y: batch[1]})
            if step % 50 == 0:
                val_loss, = sess.run([mse_loss], feed_dict={x: valid_dataset, y: valid_labels})
                print('step %d, training loss %f, val_loss %f' % (step, loss, val_loss))
                losses.append(loss)
                val_losses.append(val_loss)
        saver = tf.train.Saver()
        saver.save(sess, './models/fc_baseline.ckpt')

    ax = plt.gca()
    ax.plot(range(len(losses)), losses, label="training loss")
    ax.plot(range(len(losses)), val_losses, label="validation loss")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")

    plt.show()

if __name__ == '__main__':
    # pass
    tf.app.run(main=main, argv=[sys.argv[0]])
