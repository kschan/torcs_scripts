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
import socket, sys, getopt, os, time
PI= 3.14159265359
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
from Client import Client, ServerState, DriverAction, destringify

model = Model('cnn_steer')
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

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

    states = np.array(statesAsArray(S)).reshape([1, -1])
    states = states[:, model.states_idxs]
    # states[0,0] = 2.0*(states[0,0] - angle_min)/(angle_max-angle_min) - 1
    # states[0,1] = 2.0*(states[0,1] - trackPos_min)/(trackPos_max - trackPos_min) - 1

    steering, accel = sess.run([model.predictions_steer, model.predictions_accel], feed_dict={model.x: states})

    if step%100 == 0:
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print('distRaced:, ', S['distRaced'], 'trackPos: ', S['trackPos'], 'steering: ', steering, '\r',)
        # print('%6.1f '*19 % tuple(S['track']))
        track_pos.append(S['trackPos'])

    R['steer'] = steering
    R['accel'] = accel

    return

# ================ MAIN ================
if __name__ == "__main__":
    track_pos = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/fc_baseline.ckpt")
        C= Client(p=3001)
        for step in range(C.maxSteps,0,-1):
            C.get_servers_input()
            drive(C, sess)
            C.respond_to_server()
        C.shutdown()

    plt.plot(track_pos)
    plt.show()

