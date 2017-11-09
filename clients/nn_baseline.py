#!/usr/bin/python
# snakeoil.py
# Chris X Edwards <snakeoil@xed.ch>
# Snake Oil is a Python library for interfacing with a TORCS
# race car simulator which has been patched with the server
# extentions used in the Simulated Car Racing competitions.
# http://scr.geccocompetitions.com/
#
# To use it, you must import it and create a "drive()" function.
# This will take care of option handling and server connecting, etc.
# To see how to write your own client do something like this which is
# a complete working client:
# /-----------------------------------------------\
# |#!/usr/bin/python                              |
# |import snakeoil                                |
# |if __name__ == "__main__":                     |
# |    C= snakeoil.Client()                       |
# |    for step in xrange(C.maxSteps,0,-1):       |
# |        C.get_servers_input()                  |
# |        snakeoil.drive_example(C)              |
# |        C.respond_to_server()                  |
# |    C.shutdown()                               |
# \-----------------------------------------------/
# This should then be a full featured client. The next step is to
# replace 'snakeoil.drive_example()' with your own. There is a
# dictionary which holds various option values (see `default_options`
# variable for all the details) but you probably only need a few
# things from it. Mainly the `trackname` and `stage` are important
# when developing a strategic bot.
#
# This dictionary also contains a ServerState object
# (key=S) and a DriverAction object (key=R for response). This allows
# you to get at all the information sent by the server and to easily
# formulate your reply. These objects contain a member dictionary "d"
# (for data dictionary) which contain key value pairs based on the
# server's syntax. Therefore, you can read the following:
#    angle, curLapTime, damage, distFromStart, distRaced, focus,
#    fuel, gear, lastLapTime, opponents, racePos, rpm,
#    speedX, speedY, speedZ, track, trackPos, wheelSpinVel, z
# The syntax specifically would be something like:
#    X= o[S.d['tracPos']]
# And you can set the following:
#    accel, brake, clutch, gear, steer, focus, meta
# The syntax is:
#     o[R.d['steer']]= X
# Note that it is 'steer' and not 'steering' as described in the manual!
# All values should be sensible for their type, including lists being lists.
# See the SCR manual or http://xed.ch/help/torcs.html for details.
#
# If you just run the snakeoil.py base library itself it will implement a
# serviceable client with a demonstration drive function that is
# sufficient for getting around most tracks.
# Try `snakeoil.py --help` to get started.

# for Python3-based torcs python robot client
import socket
import sys
import getopt
import os
import time
PI= 3.14159265359

import numpy as np
import tensorflow as tf

from Client import Client, ServerState, DriverAction, destringify


num_inputs = 1
states_idxs = [0] + list(range(54, 74)) 
num_states = len(states_idxs)
x = tf.placeholder(tf.float32, [None, num_states], name="x")
y = tf.placeholder(tf.float32, [None, num_inputs], name="y")

x_norm = tf.nn.l2_normalize(x, dim = 1, epsilon=1e-12, name = 'x_norm')
# y_norm = tf.nn.l2_normalize(y, dim = 1, epsilon=1e-12, name = 'y_norm')

fc1 = tf.nn.relu(tf.layers.dense(x_norm, 256))
# fc1 = tf.nn.batch_normalization(fc1)

fc2 = tf.nn.relu(tf.layers.dense(fc1, 256))
# fc2 = tf.nn.batch_normalization(fc2)

predictions = tf.layers.dense(fc2, num_inputs, name="predictions")


def statesAsArray(S):
    res = []
    keys = ['angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'focus', \
            'fuel', 'gear', 'lastLapTime', 'opponents', 'racePos', 'rpm', \
            'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel', 'z']

    for key in keys:
        try:
            res.extend(S[key])
        except TypeError:
            res.extend([S[key]])

    return res


def drive(c, sess):

    S,R= c.S.d,c.R.d

    states = np.asarray(statesAsArray(S)).reshape([1, -1])
    states = states[:, states_idxs]
    
    inputs = sess.run(predictions, feed_dict={x: states}).flatten()

    print(inputs)


    R['steer'] = inputs[0]

    return

# ================ MAIN ================
if __name__ == "__main__":

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/fc_baseline.ckpt")
        C= Client(p=3001)
        for step in range(C.maxSteps,0,-1):
            C.get_servers_input()
            drive(C, sess)
            C.respond_to_server()
        C.shutdown()
