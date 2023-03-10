import pyglet
from game_constants import *


class Car(pyglet.sprite.Sprite):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.velocity_x, self.velocity_y, self.rotation_speed = 0.0, 0.0, 0.0

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
        self.rotation += self.rotation_speed * dt
        self.check_boundaries()
