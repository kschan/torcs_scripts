#!/usr/bin/python
# See http://xed.ch/help/torcs.html for documentation on messages

# for Python3-based torcs python robot client
import pygame
from pygame.locals import *
from sys import exit
PI= 3.14159265359

from Client import Client, ServerState, DriverAction
import time

steering_angle = 0
accel = 0
brake = 0


def getPresses():
    pressed = []
    for i in pygame.event.get():
        if i.type==QUIT:
            exit()
    if pygame.key.get_focused():
        press=pygame.key.get_pressed()
        for i in range(0,len(press)): 
            if press[i]==1:
                name=pygame.key.name(i) 
                pressed.append(name)
    # print('%s' % ', '.join(pressed))
    return pressed


def drive(c):
    pressed = getPresses()
    S,R= c.S.d,c.R.d
    global steering_angle

    steering_rate = .3

    # Steer To Corner
    # -1 is full right, 1 is full left
    if('right' in pressed):
        steering_angle -= steering_rate     
    elif('left' in pressed):
        steering_angle += steering_rate
    else:
        steering_angle = 0

    R['steer']= steering_angle

    if('up' in pressed):
        R['accel'] = 1
    else:
        R['accel'] = 0

    if('down' in pressed):
        R['brake'] = 1
    else:
        R['brake'] = 0

    # Traction Control System
    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
       R['accel']-= .2

    # Automatic Transmission
    R['gear']=1
    if S['speedX']>50:
        R['gear']=2
    if S['speedX']>80:
        R['gear']=3
    if S['speedX']>110:
        R['gear']=4
    if S['speedX']>140:
        R['gear']=5
    if S['speedX']>170:
        R['gear']=6
    return

# ================ MAIN ================
if __name__ == "__main__":
    C= Client(p=3001)
    pygame.init()
    screen=pygame.display.set_mode((640,480),0,24)
    pygame.display.set_caption("Key Press Test")
    f1=pygame.font.SysFont("comicsansms",24)
    start_time = time.time()
  
    for step in range(C.maxSteps,0,-1):
        C.get_servers_input()
        # freq = 1/(time.time()-start_time)
        # print(freq)
        # start_time = time.time()

        drive(C)
        C.respond_to_server()

    C.shutdown()
    