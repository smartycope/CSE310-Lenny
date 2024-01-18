import smbus
from math import degrees, atan2
import time
from MPU6050 import MPU6050
# from PID import PID
import motor
# import standup
from Util import dist
import os
from datetime import datetime
# from threading import Timer
from time import time as now
from math import pi
from os.path import join, dirname
from simple_pid import PID
from sshkeyboard import listen_keyboard
import threading


# Don't touch these
address = 0x68  # This is the address value read via the i2cdetect command
bus = smbus.SMBus(1)  # or bus = smbus.SMBus(1) for Revision 2 boards
CALIBRATION=0
sensor = MPU6050(bus, address, "MPU6050", calibrate=CALIBRATION)

kpVal, kiVal, kdVal = 0.96, .0, 1.0
pidDebug = False
PRINT_ANGLE = True
LOG_DATA = False
DATA_FILE = join(dirname(__file__), 'data.txt')

# This is important
# Use this link to tune the parameters:
# http://www.grauonline.de/alexwww/ardumower/pid/pid.html
# Yeah, it isn't actually as helpful as I thought...

# looks decent, with a just little bit of overcorrection
# In practice, way too much way too fast (but not jittery)
# pid=PID(2.14, .74, 1.08)

# looks pretty good, more stochastic, but smooth output with no overcorrection
# In practice, way too much way too fast (but not jittery)
# pid=PID(1.88, .28, 1.19)

# What we came with
# Actually pretty close. It feels a *little* jittery (with fresh battery), but it seems to undercorrect
# pid=PID(1.0, -0.04, 0.0)

# The default we found online
# Not terrible, but over corrects
# pid=PID(2, 0, 1)

# TOO slow, but on the right track?
# pid=PID(.4,0,1)

# Way over reacts
# pid=PID(.5,.05,1)

# WORKS!!! (for a bit)
# pid=PID(.47, .015, 1.1)

# 6s 5.5s
# pid=PID(.48, .015, 1.2)

# less (4s?)
# pid=PID(.48, .015, 1.0)

# 5s
# pid=PID(.48, .015, 1.1)

# 11s
# pid=PID(.47, .02, 1.2)

# 11.4s -- dipped a lot, but also recovered remakably
# pid=PID(.47, .028, 1.2)

# 9s -- looked sketchy
# pid=PID(.48, .028, 1.2)

# missed the time -- definite feedback loop
# pid=PID(.46, .028, 1.2)

# 30s -- definite feedback loop though, it just recovered well -- may have left the timer running, at least 15s though
# pid=PID(.47, .028, 1.1)

# almost immediate fail
# pid=PID(.47, .028, 1.0)

# 3s
# pid=PID(.47, .032, 1.15)

# 13.7s -- recovered excellently
# pid=PID(.47, .03, 1.15)

# 10s -- corrects immediately too much just a bit
# pid=PID(.48, .03, 1.15)

# 9.3s -- moved a lot
# pid=PID(.47, .025, 1.15)

# <5s
# pid=PID(.47, .03, 1.1)

# 16.3s -- moved a lot in short bursts
# pid=PID(.47, .03, 1.2)

# 5.8s
# pid=PID(.47, .03, 1.3)

# 5.1s
# pid=PID(.47, .0305, 1.18)

# 5.1s
# pid=PID(.47, .029, 1.09)

# 4.4 -- ALRIGHT, LISTEN. Buddy. Lenny. Whatcha doin there, huh?
# This is the same as on line 82
# pid=PID(.47, .03, 1.15)

#
# pid=PID(.44, .026, 1.14)

# Kind of good? Good, until it starts to fall, then dies
# pid=PID(.6, .012, .5)

# coeff=1.
# pid=PID(.73 * coeff, -.005 * coeff, .7 * coeff)

# This was decentish I think, too slow though
# pid=PID(.5, .008, 0.1)
# pid=PID(1., .07, 1)

# pid=PID(.47, .028, 1.1)

# Was remarkably closeish
# pid=PID(.95, .07, 1)

# I got this from https://pidtuner.com/
#// I think I did it right?
# iMax=15
# pid=PID(0.31910370898906526, 0.04015980668509047, 0, Integrator_max=iMax, Integrator_min=-iMax)

# pid=PID(.006, .0002, .005)
# pid=PID(1.4, .08, 0.0)
# pid=PID(.95, .07, 1)
# pid=PID(.8, .4, .00)

# This is the speed multiplier
# This is incredibly finiky
# was .055
pidMultiplier = .055
# pid.output_limits = (-100, 100)


# pid=PID(1.2, .012, 0.01)
# pid=PID(1.4, 0, 0)
# pid=PID(.6, .005, 0)
# pid=PID(.44, 3, 1.14)



# Awaiting Final Values!
# pid=PID(kpVal, kiVal, kdVal)
# pid.setpoint = 0

# The comp filter tracks its own time
dtTimer = now()

# For the comp filter
yaw = 0
pitch = 0
roll = 0
gyroRoll = 0
gyroPitch = 0
gyroYaw = 0

# For emergencies
s = motor.stop

# Because too much data is annoying
printdelay = 0
DELAY = 20

# Specific pauses for the standup function
pause1=p1=.5
pause2=p2=.3 # was .294
pause25=p25=.1
# pause3=p3=.1
# joltAngle=-.2
# checkPause=.05

# This is how close we care to be to the target angle (we think) -- actually probably not that?
# Very finiky, beware
complimenteryFilterConst = .98 # was .98


# How close to the top we need to be before we stop caring
balancedTolerance = 0.009

# How close to falling over we need to be before we just give up
motorKillTolerance = 0.1

# The face down and face up roll angles
faceUpRollVal = 40
faceDownRollVal = -40

# How many consecutive readings we need to get that we've fallen over before we believe them
killReadingsReq = 3
killReadingCount = 0

# Tread carefully
manualRollOffset = 0.0

# Erase the old data
#if LOG_DATA:
#    with open(DATA_FILE, 'w') as f:
#        f.write(f'data.append(({pid.tunings}, {CALIBRATION},  [\n')

# How often we're running the main loop (in milliseconds)
# 5 -> 10 -- dramatic improvement
loopDelay = 12


# Digital Live PID Tuner:    +- kp => U/J,    +- ki => I/K,    +- kd => O/L
# Jacob Hansen, 2021
inc = 0.01  # Increment Value
def press(key):
    global kpVal, kiVal, kdVal
    if key == 'u':
        kpVal += inc
    if key == 'j':
        kpVal += -inc
    if key == 'i':
        kiVal += inc
    if key == 'k':
        kiVal += -inc
    if key == 'o':
        kdVal += inc
    if key == 'l':
        kdVal += -inc
    if key == '0':
        motor.stop()
    if not key ==  'e': # macro used to increase responsiveness, really jank way to do it lol
      print("Kp: ", round(kpVal, 3), "\tKi: ", round(kiVal, 3),"\tKd: ", round(kdVal, 3))

def potentiometer(): #
  print("\nDigital PID Tuner: Written By Jacob Hansen")
  print("Increment Keys:\tU/J = +kP/-kP\tI/K = +kI/-kI\tO/L = +kP/-kP \nPress = to quit\n")
  print("Kp: ", round(kpVal, 3), "\tKi: ", round(kiVal, 3),"\tKd: ", round(kdVal, 3))
  
  listen_keyboard(
    on_press=press,
    until="=",
  )

# Runs a function on a loop for a specific amount of time
def psleep(seconds, func):
    end = now()
    while now() < end:
        func()

def closeEnough(a, b, tolerance):
    """ Returns True if a is within tolerance range of b """
    return a <= b + tolerance and a >= b - tolerance

def compFilter():
    global yaw, pitch, roll, dtTimer, gyroRoll, gyroPitch, gyroYaw, printdelay, DELAY, kpVal, kiVal, kdVal
        
    # Get delta time and record time for next call
    dt = now() - dtTimer
    dtTimer = now()

    sensor.read_accel_data()
    sensor.read_gyro_data()

    # Acceleration vector angle
    # accPitch = degrees(atan2(sensor.accel_scaled_y, sensor.accel_scaled_z))
    accRoll  = degrees(atan2(sensor.accel_scaled_x, sensor.accel_scaled_z)) # add PI?

    # Gyro integration angle
    # gyroRoll -= sensor.gyro_scaled_y * dt
    # gyroRoll += sensor.gyro_scaled_y * dt
    # gyroPitch += sensor.gyro_scaled_x * dt
    # gyroYaw += sensor.gyro_scaled_z * dt
    # yaw = gyroYaw

    # yaw = sensor.gyro_scaled_z * dt

    # Comp filter
    roll  = (complimenteryFilterConst)*(roll  - sensor.gyro_scaled_y * dt) + (1-complimenteryFilterConst)*(accRoll)
    # pitch = (complimenteryFilterConst)*(pitch + sensor.gyro_scaled_x * dt) + (1-complimenteryFilterConst)*(accPitch)

    # Print data
    # if not printdelay % DELAY:

        # print(" R: " + str(round(roll,1)))
        # print()
            # + " P: " + str(round(pitch,1)) \
            # + " Y: " + str(round(yaw,1)))

    # return (yaw, pitch, roll)
    return roll


###########################
#          MAIN           #
###########################
def main():
    global pidDebug
    print("####################################################")
    print("#\nLENNY-BOT 1.0! By Copeland Carter and Jacob Hansen\n#")
    print("####################################################\n")
    try:        
        if pidDebug:
          pidThread = threading.Thread(target=potentiometer, args=())
          balanceThread = threading.Thread(target=balance, args=())
          balanceThread.start()
          pidThread.start()
          pidThread.join()
          balanceThread.join()
        else:
          balance()

    # Make sure the motors stop when we kill it
    finally:
        motor.stop()
        if LOG_DATA:
            with open(DATA_FILE, 'a') as f:
                f.write(']))')

# Balance Function
def balance():
  global loopDelay, killReadingCount, kpVal, kiVal, kdVal
  start = now()
  #pid=PID(kpVal, kiVal, kdVal, setpoint = 4.53)
  while True:  
    # printdelay += 1       
    pid=PID(kpVal, kiVal, kdVal, setpoint = 2.69)
    
    # Get the adjusted values from the complimentary filter
    # adjYaw, adjPitch, adjRoll = compFilter(time_diff)
    rollRaw = compFilter()
    adjRoll = rollRaw + manualRollOffset

    # Put the adjusted roll value into the PID
    # pidRaw = pid.update(adjRoll)
    # pidRaw = pid(adjRoll)
    pidRaw = pid(rollRaw)
    speed = pidRaw * pidMultiplier  

    if LOG_DATA:
      with open(DATA_FILE, 'a') as f:
        f.write(f'{now()-start}, {rollRaw}, {pidRaw},\n')

    if PRINT_ANGLE and not printdelay % DELAY:
         print('adjRoll:', round(adjRoll, 2))
         print('speed:', round(speed, 2))
         print('pid raw:', round(speed / pidMultiplier, 2))
         print()

    # First test if we're laying down -- if so, just give up
    # if closeEnough(adjRoll, faceUpRollVal, motorKillTolerance):

    if rollRaw > faceUpRollVal:
      killReadingCount += 1  
    if killReadingCount > killReadingsReq:
      break
    elif rollRaw < faceDownRollVal:
      killReadingCount += 1
    if killReadingCount > killReadingsReq:
      break

    
    # Otherwise, move accordingly
    if speed < 0 and not closeEnough(speed, 0, balancedTolerance):
      motor.forward(abs(speed))
      #killReadingCount = 0
    elif speed > 0 and not closeEnough(speed, 0, balancedTolerance):
      motor.reverse(abs(speed))
      #killReadingCount = 0
    # We need no adjustment
    else:
      # print("We're at the top, stalling...")
      killReadingCount = 0
      # motor.stop()

    time.sleep(loopDelay / 1000)

# Not Used
def standup():
    try:
        # First get if we're face down or face up
        sensor.update()
        faceUp = sensor.roll > 0

        forward = int(not faceUp)
        backward = int(faceUp)
        rollSide = 1 if faceUp else -1

        # Using this instead of motor.raw.forward cause I think it's slightly faster for some reason
        # Just kidding, it's not, but I don't want to rewrite it
        print('Going forward')
        motor.raw.set_motors(1, forward, 1, forward)
        time.sleep(.6)

        print('Going backward')
        motor.raw.set_motors(1, backward, 1, backward)
        time.sleep(.45)

        # print('Jolting back real quick')
        # motor.raw.set_motors(1, forward, 1, forward)
        # time.sleep(pause25)

        main()
    finally:
        motor.stop()

main()