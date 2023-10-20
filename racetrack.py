import math
import pyglet as pg
from pyglet.window import mouse
from game_settings import *


def print_boundaries(bound_arr):
    for boundary in bound_arr:
        print("visible: {}, start: {}, end: ({},{})".format(boundary.visible,
                                                            boundary.position,
                                                            boundary.x2, boundary.y2))


def mouse_hit_box(x, y, mouse_x, mouse_y):
    """This method checks, if the mouse click approximately hit a point."""
    return math.sqrt((mouse_x - x)**2 + (mouse_y - y)**2) < GameSettings.MOUSE_CLICK_HIT_BOX


class Racetrack(pg.sprite.Sprite):

    IMG_WIDTH = 650
    IMG_HEIGHT = 650

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = GameSettings.WINDOW_WIDTH // 2
        self.y = GameSettings.WINDOW_HEIGHT // 2
        self.boundaries = []
        self.boundaries_start_coord = None
        self.boundary_curr_start_coord = None
        self.goals = []
        self.goal_start_coord = None
        self.n_goals = 0
        self.boundary_batch = pg.graphics.Batch()
        self.goal_batch = pg.graphics.Batch()

    def create_boundary(self, x, y, button):
        if button == mouse.RIGHT:
            # remove last boundary
            print("remove last boundary")
            if self.boundaries:
                b = self.boundaries.pop()
                self.boundary_curr_start_coord = (b.x, b.y)

        elif button == mouse.LEFT:

            if self.boundary_curr_start_coord is None:
                print("boundaries start point")
                self.boundaries_start_coord = (x, y)
                self.boundary_curr_start_coord = (x, y)

            elif mouse_hit_box(self.boundaries_start_coord[0],
                               self.boundaries_start_coord[1], x, y):
                # draw last boundary, set end point to first start point
                print("final boundary")
                b = pg.shapes.Line(self.boundary_curr_start_coord[0],
                                   self.boundary_curr_start_coord[1],
                                   self.boundaries_start_coord[0],
                                   self.boundaries_start_coord[1],
                                   color=GameSettings.BOUNDARY_COLOR,
                                   width=GameSettings.LINE_WIDTH,
                                   batch=self.boundary_batch)
                self.boundaries.append(b)
                self.boundaries_start_coord = None
                self.boundary_curr_start_coord = None

            else:
                # draw next boundary
                print("next boundary")
                if x != self.boundary_curr_start_coord[0] or y != self.boundary_curr_start_coord[1]:
                    boundary = pg.shapes.Line(self.boundary_curr_start_coord[0],
                                              self.boundary_curr_start_coord[1], x, y,
                                              color=GameSettings.BOUNDARY_COLOR,
                                              width=GameSettings.LINE_WIDTH,
                                              batch=self.boundary_batch)
                    self.boundaries.append(boundary)
                    self.boundary_curr_start_coord = (x, y)

    def create_goal(self, x, y, button):
        if button == mouse.RIGHT:
            if self.goal_start_coord is None and self.goals:
                print("remove last goal")
                self.goals.pop()
                self.n_goals -= 1
            else:
                print("remove start point")
                self.goal_start_coord = None

        if button == mouse.LEFT:
            if self.goal_start_coord is None:
                print("set start point")
                self.goal_start_coord = (x, y)
            else:
                print("next goal")
                g = pg.shapes.Line(self.goal_start_coord[0], self.goal_start_coord[1], x, y,
                                   color=GameSettings.GOAL_COLOR,
                                   width=GameSettings.LINE_WIDTH,
                                   batch=self.goal_batch)
                self.goals.append(g)
                self.n_goals += 1
                self.goal_start_coord = None

    def draw(self):
        super().draw()
        self.boundary_batch.draw()
        self.goal_batch.draw()
