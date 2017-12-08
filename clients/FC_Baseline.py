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
import socket, sys, getopt, os, time, pickle, time
from Client import Client, ServerState, DriverAction, destringify
PI = 3.14159265359

def save_dict(data_log_dict, count):
    save_name = "pid_" + str(count).zfill(7) + ".pickle"
    pickle_out = open(save_name, "wb")
    pickle.dump(data_log_dict, pickle_out)
    pickle_out.close()

def drive(c, count):
    S, R = c.S.d, c.R.d

    target_speed = 100

    # Steer To Corner
    R['steer']= S['angle']*10 / PI
    # Steer To Center
    R['steer']-= S['trackPos']*.10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer']*50):
        R['accel'] += .01
    else:
        R['accel'] -= .01

    if S['speedX']<10:
       R['accel'] += 1/(S['speedX']+.1)

    # Automatic Transmission
    R['gear']=S['gear']

    if (S['gear'] == 0):    # shift out of neutral
        R['gear'] = 1

    if (S['speedX']>70 and S['gear'] <= 1):
        R['gear']=2

    if (S['speedX']>120 and S['gear'] <= 2):
        R['gear']=3

    if (S['speedX']>160 and S['gear'] <= 3):
        R['gear']=4

    if (S['speedX']>200 and S['gear'] <= 4):
        R['gear']=5

    if (S['speedX']>240 and S['gear'] <= 5):
        R['gear']=6

    if (S['speedX']<65 and S['gear'] >= 2):
        R['gear']=1

    if (S['speedX']<115 and S['gear'] >= 3):
        R['gear']=2

    if (S['speedX']<155 and S['gear'] >= 4):
        R['gear']=3

    if (S['speedX']<195 and S['gear'] >= 5):
        R['gear']=4

    if (S['speedX']<235 and S['gear'] >= 6):
        R['gear']=5

    # Data_log
    if count %3  == 0:
        data_log_dict = {}
        data_log_dict['angle']    = S['angle']
        data_log_dict['trackPos'] = S['trackPos']
        data_log_dict['speedX']   = S['speedX']
        data_log_dict['steer']    = R['steer']
        data_log_dict['accel']    = R['accel']

        save_dict(data_log_dict, count)

    return

if __name__ == "__main__":
    C = Client(p=3001)
    count = 0
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive(C, count)
        count += 1
        C.respond_to_server()
    C.shutdown()
