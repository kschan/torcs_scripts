# Joystick controls for TORCS with datalogging.
# Built based on js_linux from https://gist.github.com/rdb/8864666

import os, struct, array, datetime
import numpy as np
from threading import Thread
from fcntl import ioctl
from sys import exit
from Client import Client

PI= 3.14159265359

class Input(object):
    def __init__(self):
        self.button_states = {}
        self.axis_states = {}

def jsRead(input):
    # Iterate over the joystick devices.
    print('Available devices:')

    for fn in os.listdir('/dev/input'):
        if fn.startswith('js'):
            print('  /dev/input/%s' % (fn))

    # These constants were borrowed from linux/input.h
    axis_names = {
        0x00 : 'x',
        0x01 : 'y',
        0x02 : 'z',
        0x03 : 'rx',
        0x04 : 'ry',
        0x05 : 'rz',
        0x06 : 'trottle',
        0x07 : 'rudder',
        0x08 : 'wheel',
        0x09 : 'gas',
        0x0a : 'brake',
        0x10 : 'hat0x',
        0x11 : 'hat0y',
        0x12 : 'hat1x',
        0x13 : 'hat1y',
        0x14 : 'hat2x',
        0x15 : 'hat2y',
        0x16 : 'hat3x',
        0x17 : 'hat3y',
        0x18 : 'pressure',
        0x19 : 'distance',
        0x1a : 'tilt_x',
        0x1b : 'tilt_y',
        0x1c : 'tool_width',
        0x20 : 'volume',
        0x28 : 'misc',
    }

    button_names = {
        0x120 : 'trigger',
        0x121 : 'thumb',
        0x122 : 'thumb2',
        0x123 : 'top',
        0x124 : 'top2',
        0x125 : 'pinkie',
        0x126 : 'base',
        0x127 : 'base2',
        0x128 : 'base3',
        0x129 : 'base4',
        0x12a : 'base5',
        0x12b : 'base6',
        0x12f : 'dead',
        0x130 : 'a',
        0x131 : 'b',
        0x132 : 'c',
        0x133 : 'x',
        0x134 : 'y',
        0x135 : 'z',
        0x136 : 'tl',
        0x137 : 'tr',
        0x138 : 'tl2',
        0x139 : 'tr2',
        0x13a : 'select',
        0x13b : 'start',
        0x13c : 'mode',
        0x13d : 'thumbl',
        0x13e : 'thumbr',

        0x220 : 'dpad_up',
        0x221 : 'dpad_down',
        0x222 : 'dpad_left',
        0x223 : 'dpad_right',

        # XBox 360 controller uses these codes.
        0x2c0 : 'dpad_left',
        0x2c1 : 'dpad_right',
        0x2c2 : 'dpad_up',
        0x2c3 : 'dpad_down',
    }

    axis_map = []
    button_map = []

    # Open the joystick device.
    fn = '/dev/input/js0'
    print('Opening %s...' % fn)
    jsdev = open(fn, 'rb')

    # Get the device name.
    #buf = bytearray(63)
    buf = array.array('B', [0] * 64)
    ioctl(jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
    js_name = buf.tostring()
    print('Device name: %s' % js_name)

    # Get number of axes and buttons.
    buf = array.array('B', [0])
    ioctl(jsdev, 0x80016a11, buf) # JSIOCGAXES
    num_axes = buf[0]

    buf = array.array('B', [0])
    ioctl(jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
    num_buttons = buf[0]

    # Get the axis map.
    buf = array.array('B', [0] * 0x40)
    ioctl(jsdev, 0x80406a32, buf) # JSIOCGAXMAP

    for axis in buf[:num_axes]:
        axis_name = axis_names.get(axis, 'unknown(0x%02x)' % axis)
        axis_map.append(axis_name)
        input.axis_states[axis_name] = 0.0

    # Get the button map.
    buf = array.array('H', [0] * 200)
    ioctl(jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

    for btn in buf[:num_buttons]:
        btn_name = button_names.get(btn, 'unknown(0x%03x)' % btn)
        button_map.append(btn_name)
        input.button_states[btn_name] = 0

    print('%d axes found: %s' % (num_axes, ', '.join(axis_map)))
    print('%d buttons found: %s' % (num_buttons, ', '.join(button_map)))

    while True:
        evbuf = jsdev.read(8)
        if evbuf:
            time, value, type, number = struct.unpack('IhBB', evbuf)

            if type & 0x01:
                button = button_map[number]
                if button:
                    input.button_states[button] = value

            if type & 0x02:
                axis = axis_map[number]
                if axis:
                    fvalue = value / 32767.0
                    input.axis_states[axis] = fvalue


def drive(c, inputs):
    S,R= c.S.d,c.R.d
    button_states = inputs.button_states
    axis_states = inputs.axis_states

    # Steer
    # -1 is full right, 1 is full left

    R['steer']= -axis_states['rx'] * .75  # js gives -1 for full left

    R['accel'] = (axis_states['rz'] + 1)/2.
    R['brake'] = ((axis_states['z'] + 1)/2.)**2*.5

    # # Traction Control System
    # if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
    #    (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
    #    R['accel']-= .2

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


    if button_states['thumbl']: # left thumb button force reverse
        R['gear'] = -1

    if button_states['tr']:     # right trigger force upshift
        R['gear'] = S['gear'] + 1
    elif button_states['tl']:   # left trigger force downshift
        R['gear'] = S['gear'] - 1

    return

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

def inputsAsArray(R):
    res = []
    keys = ['accel', 'brake', 'gear', 'steer', 'clutch', 'focus', 'meta']

    for key in keys:
        try:
            res.extend(R[key])
        except TypeError:
            res.extend([R[key]])

    return res

# ================ MAIN ================
if __name__ == "__main__":
    C= Client(p=3001)

    # We put joystick processing in a thread to avoid blocking read
    inputs = Input()
    t = Thread(target = jsRead, args=[inputs])
    t.daemon = True
    t.start()

    logged_states = np.zeros((C.maxSteps, 79))
    logged_inputs = np.zeros((C.maxSteps, 11))
    
  
    for step in range(C.maxSteps):
        response = C.get_servers_input() # this will be -1 if shutdown
        if (response == -1):
            print('CAUGHT THE SHUTDOWN AT STEP %d' % step)
            break
        drive(C, inputs)
        logged_states[step, :] = statesAsArray(C.S.d)
        logged_inputs[step, :] = inputsAsArray(C.R.d)
        C.respond_to_server()

    C.shutdown()

    # Slice unused parts of matrix
    logged_states = logged_states[:step, :]
    logged_inputs = logged_inputs[:step, :]

    now = datetime.datetime.now().strftime("%H_%M_%S_%m-%d-%Y")

    np.save("logs/logged_states_%s" % now, logged_states)
    np.save("logs/logged_inputs_%s" % now, logged_inputs)

    
