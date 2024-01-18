""" This is the interface class to the gyroscope and accelerometer """

import smbus
import math
from math import atan2
import Util as I2CUtils
from Util import dist
from time import time as now
from time import sleep

STANDARD_GRAVITY = 9.80665

class MPU6050(object):
    '''Simple MPU-6050 implementation'''
    PWR_MGMT_1 = 0x6b
    FS_SEL = 0x1b
    FS_250 = 0
    FS_500 = 1
    FS_1000 = 2
    FS_2000 = 3
    AFS_SEL = 0x1c
    AFS_2g = 0
    AFS_4g = 1
    AFS_8g = 2
    AFS_16g = 3
    ACCEL_START_BLOCK = 0x3b
    ACCEL_XOUT_H = 0
    ACCEL_XOUT_L = 1
    ACCEL_YOUT_H = 2
    ACCEL_YOUT_L = 3
    ACCEL_ZOUT_H = 4
    ACCEL_ZOUT_L = 5
    ACCEL_SCALE = {
        AFS_2g:  [ 2, 16384.0],
        AFS_4g:  [ 4, 8192.0],
        AFS_8g:  [ 8, 4096.0],
        AFS_16g: [16, 2048.0]
    }
    TEMP_START_BLOCK = 0x41
    TEMP_OUT_H = 0
    TEMP_OUT_L = 1
    GYRO_START_BLOCK = 0x43
    GYRO_XOUT_H = 0
    GYRO_XOUT_L = 1
    GYRO_YOUT_H = 2
    GYRO_YOUT_L = 3
    GYRO_ZOUT_H = 4
    GYRO_ZOUT_L = 5
    GYRO_SCALE = {
        FS_250:  [250, 131.0],
        FS_500:  [500, 65.5],
        FS_1000: [1000, 32.8],
        FS_2000: [2000, 16.4]
    }
    K = 0.98
    K1 = 1 - K

    def __init__(self, bus, address, name, calibrate=500, fs_scale=FS_250, afs_scale=AFS_2g):
        '''Constructor'''
        self.calibrationSamples = calibrate
        self._resolution = 3
        self.bus = bus
        self.address = address
        self.name = name
        self.fs_scale = fs_scale
        self.afs_scale = afs_scale
        self.raw_gyro_data = [0, 0, 0, 0, 0, 0]
        self.raw_accel_data = [0, 0, 0, 0, 0, 0]
        self.raw_temp_data = [0, 0]

        self.gyro_raw_x = 0
        self.gyro_raw_y = 0
        self.gyro_raw_z = 0
        self.gyro_scaled_x = 0
        self.gyro_scaled_y = 0
        self.gyro_scaled_z = 0

        self.raw_temp = 0
        self.scaled_temp = 0

        self.accel_raw_x = 0
        self.accel_raw_y = 0
        self.accel_raw_z = 0
        self.accel_scaled_x = 0
        self.accel_scaled_y = 0
        self.accel_scaled_z = 0

        self.pitch = 0.0
        self.roll = 0.0

        self.gyroXcal = 0
        self.gyroYcal = 0
        self.gyroZcal = 0

        # self.dtTimer = now()

        # We need to wake up the module as it starts in sleep mode
        I2CUtils.i2c_write_byte(self.bus, self.address, MPU6050.PWR_MGMT_1, 0)
        # Set the gryo resolution
        I2CUtils.i2c_write_byte(self.bus, self.address, MPU6050.FS_SEL, self.fs_scale << self._resolution)
        # Set the accelerometer resolution
        I2CUtils.i2c_write_byte(self.bus, self.address, MPU6050.AFS_SEL, self.afs_scale << self._resolution)

        self.update = self.read_raw_data
        if calibrate:
            self.calibrateGyro(self.calibrationSamples)
        else:
            self.update()
        # self.read_raw_data()

    def calibrateGyro(self, N):
        # Display message
        print("Calibrating gyro with " + str(N) + " points. Do not move!")

        # Take N readings for each coordinate and add to itself
        for _ in range(N):
            self.read_gyro_data()
            self.gyroXcal += self.gyro_raw_x
            self.gyroYcal += self.gyro_raw_y
            self.gyroZcal += self.gyro_raw_z

        # Find average offset value
        self.gyroXcal /= N
        self.gyroYcal /= N
        self.gyroZcal /= N

        # Display message and restart timer for comp filter
        print("Calibration complete:")
        print("\tX axis offset: " + str(round(self.gyroXcal,1)))
        print("\tY axis offset: " + str(round(self.gyroYcal,1)))
        print("\tZ axis offset: " + str(round(self.gyroZcal,1)) + "\n")
        sleep(.5)
        # self.dtTimer = time.time()

    def read_raw_data(self):
        ''' Read the raw data from the sensor, scale it appropriately and store for later use'''
        self.read_accel_data()
        self.read_gyro_data()
        self.read_temp_data()

        # We convert these to radians for consistency and so we can easily combine later in the filter
        #self.gyro_scaled_x = math.radians(self.gyro_raw_x / MPU6050.GYRO_SCALE[self.fs_scale][1])
        #self.gyro_scaled_y = math.radians(self.gyro_raw_y / MPU6050.GYRO_SCALE[self.fs_scale][1])
        #self.gyro_scaled_z = math.radians(self.gyro_raw_z / MPU6050.GYRO_SCALE[self.fs_scale][1])

    def get_x_rotation(self, x, y, z):
        '''Returns the rotation around the X axis in radians'''
        return atan2(y, dist(x, z))

    def get_y_rotation(self, x, y, z):
        '''Returns the rotation around the Y axis in radians'''
        return -atan2(x, dist(y, z))

    def read_gyro_data(self):
        self.raw_gyro_data = I2CUtils.i2c_read_block(self.bus, self.address, MPU6050.GYRO_START_BLOCK, 6)
        self.gyro_raw_x = I2CUtils.twos_compliment(self.raw_gyro_data[MPU6050.GYRO_XOUT_H], self.raw_gyro_data[MPU6050.GYRO_XOUT_L]) - self.gyroXcal
        self.gyro_raw_y = I2CUtils.twos_compliment(self.raw_gyro_data[MPU6050.GYRO_YOUT_H], self.raw_gyro_data[MPU6050.GYRO_YOUT_L]) - self.gyroYcal
        self.gyro_raw_z = I2CUtils.twos_compliment(self.raw_gyro_data[MPU6050.GYRO_ZOUT_H], self.raw_gyro_data[MPU6050.GYRO_ZOUT_L]) - self.gyroZcal
        self.gyro_scaled_x = self.gyro_raw_x / MPU6050.GYRO_SCALE[self.fs_scale][1]
        self.gyro_scaled_y = self.gyro_raw_y / MPU6050.GYRO_SCALE[self.fs_scale][1]
        self.gyro_scaled_z = self.gyro_raw_z / MPU6050.GYRO_SCALE[self.fs_scale][1]
        return (self.gyro_scaled_x, self.gyro_scaled_y, self.gyro_scaled_z)

    def read_accel_data(self):
        self.raw_accel_data = I2CUtils.i2c_read_block(self.bus, self.address, MPU6050.ACCEL_START_BLOCK, 6)
        self.accel_raw_x = I2CUtils.twos_compliment(self.raw_accel_data[MPU6050.ACCEL_XOUT_H], self.raw_accel_data[MPU6050.ACCEL_XOUT_L])
        self.accel_raw_y = I2CUtils.twos_compliment(self.raw_accel_data[MPU6050.ACCEL_YOUT_H], self.raw_accel_data[MPU6050.ACCEL_YOUT_L])
        self.accel_raw_z = I2CUtils.twos_compliment(self.raw_accel_data[MPU6050.ACCEL_ZOUT_H], self.raw_accel_data[MPU6050.ACCEL_ZOUT_L])
        self.accel_scaled_x = self.accel_raw_x / MPU6050.ACCEL_SCALE[self.afs_scale][1] * STANDARD_GRAVITY
        self.accel_scaled_y = self.accel_raw_y / MPU6050.ACCEL_SCALE[self.afs_scale][1] * STANDARD_GRAVITY
        self.accel_scaled_z = self.accel_raw_z / MPU6050.ACCEL_SCALE[self.afs_scale][1] * STANDARD_GRAVITY
        self.pitch = self.get_x_rotation(self.accel_scaled_x,self.accel_scaled_y,self.accel_scaled_z)
        self.roll =  self.get_y_rotation(self.accel_scaled_x,self.accel_scaled_y,self.accel_scaled_z)
        return (self.accel_scaled_x, self.accel_scaled_y, self.accel_scaled_z)

    def read_temp_data(self):
        self.raw_temp_data = I2CUtils.i2c_read_block(self.bus, self.address, MPU6050.TEMP_START_BLOCK, 2)
        self.raw_temp = I2CUtils.twos_compliment(self.raw_temp_data[MPU6050.TEMP_OUT_H], self.raw_temp_data[MPU6050.TEMP_OUT_L])
        self.scaled_temp = (self.raw_temp / 340) + 36.53
        return self.raw_temp_data

    def read_pitch(self):
        self.read_accel_data()
        return self.pitch

    def read_roll(self):
        self.read_accel_data()
        return self.roll

    def read_gyro_raw_x(self):
        self.read_gyro_data()
        return self.gyro_raw_x

    def read_gyro_raw_y(self):
        self.read_gyro_data()
        return self.gyro_raw_y

    def read_gyro_raw_z(self):
        self.read_gyro_data()
        return self.gyro_raw_z

    def read_gyro_scaled_x(self):
        self.read_gyro_data()
        return self.gyro_scaled_x

    def read_gyro_scaled_y(self):
        self.read_gyro_data()
        return self.gyro_scaled_y

    def read_gyro_scaled_z(self):
        self.read_gyro_data()
        return self.gyro_scaled_z

    def read_accel_raw_x(self):
        self.read_accel_data()
        return self.accel_raw_x

    def read_accel_raw_y(self):
        self.read_accel_data()
        return self.accel_raw_y

    def read_accel_raw_z(self):
        self.read_accel_data()
        return self.accel_raw_z

    def read_accel_scaled_x(self):
        self.read_accel_data()
        return self.accel_scaled_x

    def read_accel_scaled_y(self):
        self.read_accel_data()
        return self.accel_scaled_y

    def read_accel_scaled_z(self):
        self.read_accel_data()
        return self.accel_scaled_z

    def read_raw_temp(self):
        self.read_temp_data()
        return self.raw_temp

    def read_scaled_temp(self):
        self.read_temp_data()
        return self.scaled_temp
