import math
import pyglet
from pyglet.window import key
from game_constants import *


class Car(pyglet.sprite.Sprite):

    CAR_START_POSITION_X = 165
    CAR_START_POSITION_Y = 320
    IMG_WIDTH = 16
    IMG_HEIGHT = 32
    ROTATION_SPEED = 200.0
    THRUST = 200.0

    def __init__(self, x_pos=CAR_START_POSITION_X, y_pos=CAR_START_POSITION_Y, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = x_pos
        self.y = y_pos
        self.velocity_x, self.velocity_y = 0.0, 0.0
        self.keys = dict(left=False, right=False, up=False, down=False)

    def check_boundaries(self):
        min_x = self.image.width // 2
        min_y = self.image.height // 2
        max_x = WINDOW_WIDTH - self.image.width // 2
        max_y = WINDOW_HEIGHT - self.image.height // 2

        if self.x < min_x:
            self.x = min_x
        elif self.x > max_x:
            self.x = max_x
        if self.y < min_y:
            self.y = min_y
        elif self.y > max_y:
            self.y = max_y

    def update(self, dt):
        """This method should be called at least once per frame."""
        # Update position and rotation
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        self.check_boundaries()

        if self.keys['left']:
            self.rotation -= self.ROTATION_SPEED * dt
        if self.keys['right']:
            self.rotation += self.ROTATION_SPEED * dt
        if self.keys['up']:
            rad = math.radians(self.rotation)
            x_force = math.sin(rad) * self.THRUST
            y_force = math.cos(rad) * self.THRUST
            self.velocity_x += x_force * dt
            self.velocity_y += y_force * dt
        if self.keys['down']:
            rad = math.radians(self.rotation)
            x_force = math.sin(rad) * self.THRUST
            y_force = math.cos(rad) * self.THRUST
            self.velocity_x -= x_force * dt
            self.velocity_y -= y_force * dt

    def on_key_press(self, symbol, modifiers):
        if symbol == key.UP:
            self.keys['up'] = True
        elif symbol == key.DOWN:
            self.keys['down'] = True
        elif symbol == key.LEFT:
            self.keys['left'] = True
        elif symbol == key.RIGHT:
            self.keys['right'] = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.UP:
            self.keys['up'] = False
        elif symbol == key.DOWN:
            self.keys['down'] = False
        elif symbol == key.LEFT:
            self.keys['left'] = False
        elif symbol == key.RIGHT:
            self.keys['right'] = False

