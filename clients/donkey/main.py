import glob, sys, pickle, socket, sys, getopt, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
from util import *
from Client import Client, ServerState, DriverAction, destringify

INPUT          = ['angle', 'trackPos', 'speedX']
OUTPUT         = ['steer', 'accel']
PID_PATH       = "pid_label/*pickle"
BATCH_SIZE     = 32
TRAIN_MAX_ITER = 30000

def train():
    model = Model('steer_accel')

    # Load dataset
    # # train_dataset, train_labels, \
    # # valid_dataset, valid_labels, \
    # # test_dataset , test_labels = load_human_data_pid_labels(INPUT, OUTPUT)

    train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset , test_labels = load_data_pid(INPUT, OUTPUT, PID_PATH)
    print('[ INFO ] Train dataset size:', train_dataset.shape)

    # Train begins
    losses = []; train_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for train_iter in range(TRAIN_MAX_ITER):
            print('train_iter:', train_iter)
            if train_count*BATCH_SIZE >= train_dataset.shape[0]:
                train_count = 0

            # Get batch data
            batch = get_next_batch(train_dataset, train_labels, train_count, BATCH_SIZE)
            train_steer = np.concatenate([batch[0][:, 0].reshape((-1, 1)),
                                          batch[0][:, 1].reshape((-1, 1))], axis = 1) # angle, trackPos
            label_steer = batch[1][:, 0].reshape((-1, 1)) # steer

            train_accel = np.concatenate([batch[0][:, 0].reshape((-1, 1)),
                                          batch[0][:, 1].reshape((-1, 1)),
                                          batch[0][:, 2].reshape((-1, 1))], axis = 1) # angle, trackPos, speedX
            label_accel = batch[1][:, 1].reshape((-1, 1)) # accel

            _, loss = sess.run([model.opt, model.total_loss], 
                               feed_dict={model.x_steer: train_steer, model.y_steer: label_steer,
                                          model.x_accel: train_accel, model.y_accel: label_accel})
            losses.append(loss)

            train_count += 1

        saver = tf.train.Saver()
        saver.save(sess, './weights/steer_accel.ckpt')

    # Show loss in graph
    ax = plt.gca()
    ax.plot(losses, label="loss")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()

def drive(c, sess, model):
    S, R = c.S.d, c.R.d

    # steer model
    data_steer = np.zeros((1, 2), dtype=np.float64)
    data_steer[0, 0] = normalize(S['angle'], 'angle')
    data_steer[0, 1] = normalize(S['trackPos'], 'trackPos')

    data_accel = np.zeros((1, 3), dtype=np.float64)
    data_accel[0, 0] = normalize(S['angle'], 'angle')
    data_accel[0, 1] = normalize(S['trackPos'], 'trackPos')
    data_accel[0, 2] = normalize(S['speedX'], 'speedX')

    pred_steer, pred_accel = sess.run([model.pred_steer, model.pred_accel], 
                                      feed_dict={model.x_steer: data_steer, model.x_accel: data_accel, })
    print(pred_steer, pred_accel)

    R['steer'] = pred_steer[0]
    R['accel'] = pred_accel[0]

    return

def test():
    # Build model
    model = Model('steer_accel')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./weights/steer_accel.ckpt")

        C = Client(p=3001)
        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()
            drive(C, sess, model)
            C.respond_to_server()
        C.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 1:
        raise RuntimeError("[ ERROR ] Please specify options (train/test)")

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        raise RuntimeError("[ ERROR ] Not available option")
