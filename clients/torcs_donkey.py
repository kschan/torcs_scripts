from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model

def concatenate_training_data():
    logged_input_files = glob.glob("../good_logs/logged_inputs*")
    logged_inputs = np.concatenate([np.load(file) for file in logged_input_files], axis = 0)
    print("[ INFO ] Logged_inputs: ", logged_inputs.shape)

    logged_state_files = glob.glob("../good_logs/logged_states*")
    logged_states = np.concatenate([np.load(file) for file in logged_state_files], axis = 0)
    print("[ INFO ] Logged_states: ", logged_states.shape)

    return logged_states, logged_inputs

def get_next_batch(train_dataset, train_labels, batch_size):
    batch_indices = np.random.randint(train_labels.shape[0], size=batch_size)
    batch_dataset = train_dataset[batch_indices, :]
    batch_labels = train_labels[batch_indices, :]

    return batch_dataset, batch_labels

def load_data(model):
    logged_states, logged_inputs = concatenate_training_data()
    num_samples    = logged_states.shape[0]
    random_indices = np.random.permutation(num_samples)

    train_indices  = random_indices[:(num_samples - 20000)]
    train_dataset  = logged_states[train_indices, :]
    train_dataset  = train_dataset[:, model.states_idxs]
    train_labels   = logged_inputs[train_indices, :]
    train_labels   = train_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

    valid_indices  = random_indices[(num_samples-20000):(num_samples-10000)]
    valid_dataset  = logged_states[valid_indices, :]
    valid_dataset  = valid_dataset[:, model.states_idxs]
    valid_labels   = logged_inputs[valid_indices, :]
    valid_labels   = valid_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

    test_indices   = random_indices[(num_samples-10000):]
    test_dataset   = logged_states[test_indices, :]
    test_dataset   = test_dataset[:, model.states_idxs]
    test_labels    = logged_inputs[test_indices, :]
    test_labels    = test_labels[:, model.input_idxs].reshape((-1, model.num_inputs))

    return train_dataset, train_labels, \
           valid_dataset, valid_labels, \
           test_dataset , test_labels

def train():
    model = Model('fc_steer') # fc_steer or cnn_steer

    # Load dataset
    train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset , test_labels = load_data(model)

    # Train begins
    batch_size     = 8
    train_max_iter = 50000

    train_losses, val_losses = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_max_iter):
            batch = get_next_batch(train_dataset, train_labels, batch_size)
            train_loss = sess.run([model.total_loss], feed_dict={model.x: batch[0], model.y: batch[1]})[0]
            if step % 50 == 0:
                val_loss, = sess.run([model.total_loss], feed_dict={model.x: valid_dataset, model.y: valid_labels})
                print('[ STATUS ] step %d, training loss %f, val_loss %f' % (step, train_loss, val_loss))

                train_losses.append(train_loss)
                val_losses.append(val_loss)
        saver = tf.train.Saver()
        saver.save(sess, './models/fc_baseline.ckpt')

    ax = plt.gca()
    ax.plot(range(len(train_losses)), train_losses, label="training loss")
    ax.plot(range(len(train_losses)), val_losses, label="validation loss")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")

    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        todo = 1
    else:
        raise RuntimeError("[ ERROR ] Not available option")
