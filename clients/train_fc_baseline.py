from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import sys
import numpy as np
import tempfile
from model import Model

import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *


def concatenate_training_data():
    logged_input_files = glob.glob("../good_logs/logged_inputs*")
    print(logged_input_files)
    logged_inputs = np.concatenate([np.load(file) for file in logged_input_files], axis = 0)
    print("logged_inputs: ", logged_inputs.shape)

    logged_state_files = glob.glob("../good_logs/logged_states*")
    logged_states = np.concatenate([np.load(file) for file in logged_state_files], axis = 0)
    print("logged_states: ", logged_states.shape)

    return logged_states, logged_inputs

def get_next_batch(train_dataset, train_labels, batch_size):
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

    model = Model('fc_steer')

    logged_states, logged_inputs = concatenate_training_data()
    num_samples    = logged_states.shape[0]
    random_indices = np.random.permutation(num_samples)
    train_indices  = random_indices[:(num_samples - 20000)]
    valid_indices  = random_indices[(num_samples-20000):(num_samples-10000)]
    test_indices   = random_indices[(num_samples-10000):]

    train_dataset  = logged_states[train_indices, :]
    train_dataset  = train_dataset[:, model.states_idxs]

    # print(logged_states.shape)
    pid_inputs = get_pid_inputs(logged_states)
    # print(pid_inputs.shape)
    train_dataset = logged_states[train_indices, :]
    train_labels = pid_inputs[train_indices, :]
    valid_dataset = logged_states[valid_indices,:]
    valid_labels = pid_inputs[valid_indices, :]

    train_dataset = train_dataset[:, [0, 73]]
    valid_dataset = valid_dataset[:, [0, 73]]
    # train_labels = train_labels[:,[0]]

    for input_idx in range(2):
        max_val = np.max(train_dataset[:, input_idx])
        min_val = np.min(train_dataset[:, input_idx])
        print(max_val)
        print(min_val)
        train_dataset[:, input_idx] = 2.0*(train_dataset[:, input_idx] - min_val)/(float(max_val - min_val)) - 1
        valid_dataset[:, input_idx] = 2.0*(valid_dataset[:, input_idx] - min_val)/(float(max_val - min_val)) - 1

    print(train_dataset.shape)
    print(train_labels.shape)




#     train_labels   = logged_inputs[train_indices, :]
#     train_labels   = train_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

#     valid_dataset  = logged_states[valid_indices, :]
#     valid_dataset  = valid_dataset[:, model.states_idxs]

#     valid_labels   = logged_inputs[valid_indices, :]
#     valid_labels   = valid_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

#     test_dataset   = logged_states[test_indices, :]
#     test_dataset   = test_dataset[:, model.states_idxs]

#     test_labels    = logged_inputs[test_indices, :]
#     test_labels    = test_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

    batch_size = 32

    losses, val_losses = [], []

#     # plt.hist(logged_states[:, 63], bins = 20)
#     # plt.show()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(50000):
            batch = get_next_batch(train_dataset, train_labels, batch_size)
#             # print(batch[1])
            _, loss = sess.run([model.train_step, model.total_loss], feed_dict={model.x: batch[0], model.y: batch[1]})
            if step % 50 == 0:
                val_loss, = sess.run([model.total_loss], feed_dict={model.x: valid_dataset, model.y: valid_labels})
                print('step %d, training loss %f, ' % (step, loss))
                if loss < 0.03:
                    break
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
    tf.app.run(main=main, argv=[sys.argv[0]])
