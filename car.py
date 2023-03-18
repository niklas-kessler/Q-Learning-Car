import math
import pyglet
from pyglet.window import key
from game_constants import *


class Car(pyglet.sprite.Sprite):

    CAR_START_POSITION_X = 165
    CAR_START_POSITION_Y = 320
    IMG_WIDTH = 12
    IMG_HEIGHT = 24
    MAX_VELOCITY = 200
    MIN_VELOCITY = -MAX_VELOCITY
    THRUST = 350.0
    FRICTION_DELAY = 0.6
    ROTATION_SPEED = 200.0

    def __init__(self, x_pos=CAR_START_POSITION_X, y_pos=CAR_START_POSITION_Y, user_controls=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = x_pos
        self.y = y_pos
        self.velocity = 0.0
        self.keys = dict(left=False, right=False, up=False, down=False)
        self.user_controls = user_controls

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

    def calc_velocity(self, dt):
        # natural deceleration
        self.velocity -= (self.FRICTION_DELAY * self.velocity) * dt

        # split up velocity in x and y
        rad = math.radians(self.rotation)
        velocity_x = math.sin(rad) * self.velocity
        velocity_y = math.cos(rad) * self.velocity
        return velocity_x, velocity_y

    def update(self, dt):
        """This method should be called at least once per frame."""
        # Update position and rotation
        velocity_x, velocity_y = self.calc_velocity(dt)
        self.x += velocity_x * dt
        self.y += velocity_y * dt
        self.check_boundaries()

        if self.keys['left']:
            self.rotation -= self.ROTATION_SPEED * dt
        if self.keys['right']:
            self.rotation += self.ROTATION_SPEED * dt
        if self.keys['up']:
            if self.velocity < self.MAX_VELOCITY:
                self.velocity += self.THRUST * dt
        if self.keys['down']:
            if self.velocity > self.MIN_VELOCITY:
                self.velocity -= self.THRUST * dt

    # User controls
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

