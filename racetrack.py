import pyglet as pg
from pyglet.window import mouse

from game_settings import *


def print_boundaries(bound_arr):
    for boundary in bound_arr:
        print("visible: {}, start: {}, end: ({},{})".format(boundary.visible,
                                                            boundary.position,
                                                            boundary.x2, boundary.y2))


class Racetrack(pg.sprite.Sprite):

    IMG_WIDTH = 650
    IMG_HEIGHT = 650

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = GameSettings.WINDOW_WIDTH // 2
        self.y = GameSettings.WINDOW_HEIGHT // 2
        self.boundaries = []
        self.boundary_start_coord = None
        self.racetrack_batch = pg.graphics.Batch()

    def on_mouse_press(self, x, y, button, modifiers):
        if self.boundary_start_coord is None:
            print("draw_boundaries (1/2)")
            self.boundary_start_coord = (x, y)
        else:
            if button == mouse.LEFT:
                print("draw_boundaries (2/2)")
                boundary = pg.shapes.Line(self.boundary_start_coord[0],
                                          self.boundary_start_coord[1], x, y,
                                          color=GameSettings.BOUNDARY_COLOR,
                                          batch=self.racetrack_batch)
                self.boundaries.append(boundary)
                print_boundaries(self.boundaries)
            self.boundary_start_coord = None
