import math
from math import pi, degrees, radians, atan2
import time
from time import sleep
from MPU6050 import *
from PID import PID
import motor
from time import time as now
from Util import dist

# IP=192.168.43.207; sshpass -p "raspberry" scp -r /home/** USER **/hello/class/ECEN-361-Project/From\ Lenny/LennysCode/ "pi@$IP:/home/pi/ECEN-361-Project/ECEN-361-Project/From\ Lenny/" && sshpass -p "raspberry" ssh pi@$IP

pidMultiplier = 30

# These aren't used?
# gyro_scale = 131.0
# accel_scale = 16384.0

RAD_TO_DEG = 57.29578

# Not sure what these are... when you figure out change the name to make more sense please
complimenteryFilterConst = 0.98

time_diff = 0.01

accAngX = 0.0

# Final Angle(s)
CFangleX = 0.0
CFangleX1 = 0.0

# FIX = -12.89

# Roll is forward and backward (the way the wheels turn)
# Pitch is side to side (if lenny were to hop wheel to wheel like he had to pee)
# We don't have yaw (which direction Lenny is facing when standing up)
# Roll is x
# pitch is y
# yaw is z


pause1=p1=.8
pause2=p2=.294
pause25=p25=.1
pause3=p3=.1
joltAngle=-.2
checkPause=.05

yaw = 0
pitch = 0
roll = 0

gyroRoll = 0
gyroPitch = 0
gyroYaw = 0

dtTimer = now()

gravity = 9.80665

s=motor.stop

def showRoll():
    print(f'Roll angle at {round(sensor.read_roll(), 2)}')

def psleep(seconds, func):
    end = now()
    while now() < end:
        func()

def rollAngle():
    global prevRollTime
    sensor.update()
    # ans = degrees(atan2(sensor.accel_scaled_x, sensor.accel_scaled_y) + pi)
    # ans = degrees(atan2(sensor.accel_scaled_y, dist(sensor.accel_scaled_x, sensor.accel_scaled_z)))
    ans = sensor.gyro_scaled_x * prevRollTime
    prevRollTime = now()
    print(f'Roll angle at {ans}')
    return ans


def compFilter():
    global yaw, pitch, roll, dtTimer, gyroRoll, gyroPitch, gyroYaw
    # Get delta time and record time for next call
    dt = now() - dtTimer
    dtTimer = now()

    # Acceleration vector angle
    accPitch = degrees(atan2(sensor.accel_scaled_y, sensor.accel_scaled_z))
    accRoll = degrees(atan2(sensor.accel_scaled_x, sensor.accel_scaled_z))

    # Gyro integration angle
    # gyroRoll -= sensor.gyro_scaled_y * dt
    # gyroPitch += sensor.gyro_scaled_x * dt
    gyroYaw += sensor.gyro_scaled_z * dt
    yaw = gyroYaw

    # Comp filter
    roll  = (gravity)*(roll - sensor.gyro_scaled_y*dt) + (1-gravity)*(accRoll)
    pitch = (gravity)*(pitch + sensor.gyro_scaled_x*dt) + (1-gravity)*(accPitch)

    # Print data
    print(" R: " + str(round(roll,1)) \
        + " P: " + str(round(pitch,1)) \
        + " Y: " + str(round(yaw,1)))

    return (yaw, pitch, roll)


def standUp():
    # First get if we're face down or face up
    sensor.update()
    faceUp = sensor.roll > 0

    forward = int(not faceUp)
    backward = int(faceUp)
    rollSide = 1 if faceUp else -1

    # Using this instead of motor.raw.forward cause I think it's
    # slightly faster for some reason
    print('Going forward')
    motor.raw.set_motors(1, forward, 1, forward)
    psleep(pause1, compFilter)

    print('Going backward')
    motor.raw.set_motors(1, backward, 1, backward)
    psleep(pause2, compFilter)

    print('Checking angle')
    # If we're still leaning off to the side, keep going
    while compFilter()[2] > joltAngle * rollSide:
        pass
        # psleep(checkPause, compFilter)
    psleep(pause25, compFilter)

    print('Jolting forward again')
    motor.raw.set_motors(1, forward, 1, forward)
    psleep(pause3, compFilter)

    print('Done!')
    motor.stop()

    # motor.raw.set_motors(1, forward, 1, forward); sleep(pause1); motor.raw.set_motors(1, backward, 1, backward); sleep(pause2);  showRoll(); motor.raw.set_motors(1, forward, 1, forward); sleep(pause3); motor.stop()