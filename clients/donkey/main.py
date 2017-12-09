import glob, sys, pickle, socket, sys, getopt, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
from util import load_data_pid, get_next_batch, normalize
from Client import Client, ServerState, DriverAction, destringify

MODEL_OPTION   = 'fc'
INPUT          = ['angle', 'trackPos', 'speedX']
OUTPUT         = ['steer']
PID_PATH       = "pid_log/v2/*pickle"
BATCH_SIZE     = 16
TRAIN_MAX_ITER = 10000

def train():
    # Build model
    model = Model(MODEL_OPTION, INPUT, OUTPUT)

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
        saver = tf.train.Saver()
        saver.save(sess, './weights/fc_baseline.ckpt')

    # Show loss in graph
    ax = plt.gca()
    ax.plot(range(len(train_losses)), train_losses, label="training loss")
    # ax.plot(range(len(train_losses)), val_losses, label="validation loss")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss")

    plt.show()

def drive(c, sess, model):
    S, R = c.S.d, c.R.d

    # Normalize data
    input_data = np.zeros((1,len(INPUT)), dtype=np.float64)
    for input_idx, input_val in enumerate(INPUT):
        input_data[0, input_idx] = normalize(S[input_val], input_val)

    outputs = sess.run(model.predictions, feed_dict={model.x: input_data}).flatten()

    for output_idx, output_val in enumerate(OUTPUT):
        R[output_val] = outputs[output_idx]

    return

def test():
    # Build model
    model = Model(MODEL_OPTION, INPUT, OUTPUT)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./weights/fc_baseline.ckpt")
        C = Client(p=3001)
        for step in range(C.maxSteps,0,-1):
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
