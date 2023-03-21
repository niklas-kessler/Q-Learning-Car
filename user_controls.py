import pyglet as pg
from pyglet.window import key
"""
This file contains the event-handler-methods that handle user control of the car.
"""


def on_key_press(car, symbol, modifiers):
    if symbol == key.UP:
        car.keys['up'] = True
    elif symbol == key.DOWN:
        car.keys['down'] = True
    elif symbol == key.LEFT:
        car.keys['left'] = True
    elif symbol == key.RIGHT:
        car.keys['right'] = True


def on_key_release(car, symbol, modifiers):
    if symbol == key.UP:
        car.keys['up'] = False
    elif symbol == key.DOWN:
        car.keys['down'] = False
    elif symbol == key.LEFT:
        car.keys['left'] = False
    elif symbol == key.RIGHT:
        car.keys['right'] = False
