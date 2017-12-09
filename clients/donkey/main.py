import glob, sys, pickle, socket, sys, getopt, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
from util import load_data_pid, get_next_batch

INPUT          = ['angle', 'trackPos', 'speedX']
OUTPUT         = ['steer']
PID_PATH       = "pid_log/v2/*pickle"
BATCH_SIZE     = 16
TRAIN_MAX_ITER = 10000

def train():
    # Build model
    model = Model('fc', INPUT, OUTPUT)

    # Load dataset
    train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset , test_labels = load_data_pid(model, INPUT, OUTPUT, PID_PATH)

    # Train begins
    train_losses, val_losses = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(TRAIN_MAX_ITER):
            batch = get_next_batch(train_dataset, train_labels, BATCH_SIZE)
            train_loss = sess.run([model.total_loss], feed_dict={model.x: batch[0], model.y: batch[1]})[0]
            train_losses.append(train_loss)
            # if step % 50 == 0:
            #     val_loss, = sess.run([model.total_loss], feed_dict={model.x: valid_dataset, model.y: valid_labels})
            #     print('[ STATUS ] step %d, training loss %f, val_loss %f' % (step, train_loss, val_loss))

            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
        saver = tf.train.Saver()
        saver.save(sess, './models/fc_baseline.ckpt')

    ax = plt.gca()
    ax.plot(range(len(train_losses)), train_losses, label="training loss")
    # ax.plot(range(len(train_losses)), val_losses, label="validation loss")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")

    plt.show()

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
    if len(sys.argv) < 1:
        raise RuntimeError("[ ERROR ] Please specify options (train/test)")

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        raise RuntimeError("[ ERROR ] Not available option")
