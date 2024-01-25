
from sshkeyboard import listen_keyboard
import asyncio

kpVal, kiVal, kdVal = .8, .4, .0

async def press(key):
    global kpVal, kiVal, kdVal
    if key == 'u':
        kpVal += .01
    if key == 'j':
        kpVal += -.01
    if key == 'i':
        kiVal += .01
    if key == 'k':
        kiVal += -.01
    if key == 'o':
        kdVal += .01
    if key == 'l':
        kdVal += -.01
    if not key ==  '`': # macro used to increase responsiveness, really hacky way to do it lol
      print("Kp: ", round(kpVal, 3), "\tKi: ", round(kiVal, 3),"\tKd: ", round(kdVal, 3))

def potentiometer():
  print("\nPress = to quit")
  print("Kp: ", round(kpVal, 3), "\tKi: ", round(kiVal, 3),"\tKd: ", round(kdVal, 3))
  
  listen_keyboard(
    on_press=press,
    until="=",
  )

#potentiometer()
  