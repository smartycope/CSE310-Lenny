#!/usr/bin/python
from rrb3 import *
import time

rr = RRB3(11, 11)

rr.set_motors(1, 0, 1, 0)
time.sleep(1)
rr.set_motors(0.5, 0, 0.5, 0)
time.sleep(1)
rr.set_motors(0.5, 1, 0.5, 1)
time.sleep(1)
rr.set_motors(0, 0, 0, 0)
