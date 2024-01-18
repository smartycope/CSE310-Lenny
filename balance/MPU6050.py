#!/usr/bin/python

import sys
import smbus
import math
import time
import datetime

powerMgmt1 = 0x6b

def read_byte(adr):
    return bus.read_byte_data(address, adr)

def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr+1)
    val = (high << 8) + low
    return val

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def dist(a,b):
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)

bus = smbus.SMBus(1)
address = 0x68
bus.write_byte_data(address, powerMgmt1, 0)

start = datetime.datetime.now()
end = start+datetime.timedelta(seconds=60)
print(start)
now = datetime.datetime.now()
while 1:
    gyroXout = read_word_2c(0x43)
    gyroYout = read_word_2c(0x45)
    gyroZout = read_word_2c(0x47)

    accelXout = read_word_2c(0x3b)
    accelYout = read_word_2c(0x3d)
    accelZout = read_word_2c(0x3f)

#    print datetime.datetime.now()-start
#    print accelXout, accelYout, accelZout
    print(gyroXout, gyroYout, gyroZout)
    time.sleep(.1)
    if end-datetime.datetime.now()<datetime.timedelta(seconds=0):
        sys.exit(0)
