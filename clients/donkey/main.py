from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, sys, pickle, socket, sys, getopt, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
from Client import Client, ServerState, DriverAction, destringify

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
    train_labels   = train_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    valid_indices  = random_indices[(num_samples-20000):(num_samples-10000)]
    valid_dataset  = logged_states[valid_indices, :]
    valid_dataset  = valid_dataset[:, model.states_idxs]
    valid_labels   = logged_inputs[valid_indices, :]
    valid_labels   = valid_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    test_indices   = random_indices[(num_samples-10000):]
    test_dataset   = logged_states[test_indices, :]
    test_dataset   = test_dataset[:, model.states_idxs]
    test_labels    = logged_inputs[test_indices, :]
    test_labels    = test_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    return train_dataset, train_labels, \
           valid_dataset, valid_labels, \
           test_dataset , test_labels

def read_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data_pid(model):
    # Initialize
    angle_data     = []
    track_pos_data = []
    speed_x_data   = []
    steer_data     = []
    accel_data     = []

    # Read data
    dict_paths = glob.glob("pid_log/v2/*pickle")
    for dict_path in sorted(dict_paths):
        dict_file = read_dict(dict_path)

        angle_data.append(dict_file['angle'])
        track_pos_data.append(dict_file['trackPos'])
        speed_x_data.append(dict_file['speedX'])
        steer_data.append(dict_file['steer'])
        accel_data.append(dict_file['accel'])

    # Concat data
    train_dataset = np.concatenate((np.asarray(angle_data).reshape((-1,1)), 
                                    np.asarray(track_pos_data).reshape((-1,1)), 
                                    np.asarray(speed_x_data).reshape((-1,1))), axis=1)
    train_label   = np.concatenate((np.asarray(steer_data).reshape((-1,1)),
                                    np.asarray(accel_data).reshape((-1,1))), axis=1)

    return train_dataset, train_label, None, None, None, None

def train():
    model = Model('donkey_steer') # fc_steer or cnn_steer

    # Load dataset
    train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset , test_labels = load_data(model)

    train_dataset_pid, train_labels_pid, \
    valid_dataset_pid, valid_labels_pid, \
    test_dataset_pid,  test_labels_pid = load_data_pid(model)

    train_dataset = train_dataset_pid
    train_labels  = train_labels_pid

    # Train begins
    batch_size     = 8
    train_max_iter = 10000

    train_losses, val_losses = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_max_iter):
            batch = get_next_batch(train_dataset, train_labels, batch_size)
            train_loss = sess.run([model.total_loss], feed_dict={model.x: batch[0], model.y: batch[1]})[0]
            print(train_loss)
            # if step % 50 == 0:
            #     val_loss, = sess.run([model.total_loss], feed_dict={model.x: valid_dataset, model.y: valid_labels})
            #     print('[ STATUS ] step %d, training loss %f, val_loss %f' % (step, train_loss, val_loss))

            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
        saver = tf.train.Saver()
        saver.save(sess, './models/fc_baseline.ckpt')

    # ax = plt.gca()
    # ax.plot(range(len(train_losses)), train_losses, label="training loss")
    # ax.plot(range(len(train_losses)), val_losses, label="validation loss")
    # ax.set_xlabel("training steps")
    # ax.set_ylabel("loss")

    # plt.show()

def statesAsArray(S):
    res = []
    keys = ['angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'focus', \
            'fuel', 'gear', 'lastLapTime', 'opponents', 'racePos', 'rpm', \
            'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel', 'z']

    for key in keys:
        # print(key, len(res))
        try:
            res.extend(S[key])
        except TypeError:
            res.extend([S[key]])

    return res

def drive(c, sess):
    S,R= c.S.d,c.R.d
    # statesAsArray(S)

    states = np.asarray(statesAsArray(S)).reshape([1, -1])
    states = states[:, model.states_idxs]
    
    inputs = sess.run(model.predictions, feed_dict={model.x: states}).flatten()

    print(inputs)

    if step%100 == 0:
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print('distRaced:, ', S['distRaced'], 'trackPos: ', S['trackPos'], '\r',)
        track_pos.append(S['trackPos'])

    R['steer'] = inputs[0]
    R['accel'] = inputs[1]

    return

def test():
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/fc_baseline.ckpt")
        C = Client(p=3001)
        for step in range(C.maxSteps,0,-1):
            C.get_servers_input()
            drive(C, sess)
            C.respond_to_server()
        C.shutdown()

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        raise RuntimeError("[ ERROR ] Not available option")
