import math
import pyglet as pg
from racetrack import Racetrack
from game_settings import *
from utils import *


class Car(pg.sprite.Sprite):

    CAR_START_POSITION_X = 165
    CAR_START_POSITION_Y = 320
    IMG_WIDTH = 12
    IMG_HEIGHT = 24
    MAX_VELOCITY = 200
    MIN_VELOCITY = -MAX_VELOCITY
    THRUST = 350.0
    FRICTION_DELAY = 0.6
    ROTATION_SPEED = 200.0

    def __init__(self, racetrack: Racetrack, x=CAR_START_POSITION_X, y=CAR_START_POSITION_Y,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = x
        self.y = y
        self.velocity = 0.0
        self.keys = dict(left=False, right=False, up=False, down=False)
        self.batch = pg.graphics.Batch()

        # "l" ~ left, "f" ~ front, "r" ~ right, "b" ~ back;   order: f, fr, r, br, b, bl, l, fl
        self.sensors = []
        self.intersection_points = []
        self.sensor_val = [0, 0, 0, 0, 0, 0, 0, 0]  # dict(f=0.0, fr=0.0, l=0.0, r=0.0, bl=0.0, b=0.0, br=0.0, fl=0.0,)
        self.i_goals = 0
        self.i_rounds = 0
        self.distance_next_goal = math.inf
        self.racetrack = racetrack
        self.update_sensors(init=True)

    def check_boundaries(self):
        min_x = self.image.width // 2
        min_y = self.image.height // 2
        max_x = GameSettings.WINDOW_WIDTH - self.image.width // 2
        max_y = GameSettings.WINDOW_HEIGHT - self.image.height // 2

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

    def update_obj(self, dt):
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

        self.update_sensors()
        self.sensor_intersections()

    def reset(self):
        self.x = self.CAR_START_POSITION_X
        self.y = self.CAR_START_POSITION_Y
        self.velocity = 0.0
        self.rotation = 0

    def update_sensors(self, init=False):

        w = self.width
        h = self.height
        r = math.sqrt(w**2 + h**2) / 2 # length of half diagonal
        s = GameSettings.SENSOR_LENGTH

        alpha = math.radians(self.rotation)
        beta = math.atan(w/h)  # angle between vertical (=alpha) and diagonal

        # order: save (x,y,x2,y2)-tupel; clockwise, beginning with f ending with fl
        # (f_x, f_y, f_x2, f_y2) -> (fr_x, fr_y, fr_x2, fr_y2) -> ... -> (fl_x, fl_y, fl_x2, fl_y2)
        coords = []

        start_x = self.x + math.sin(alpha) * h/2
        start_y = self.y + math.cos(alpha) * h/2
        end_x = start_x + math.sin(alpha) * s
        end_y = start_y + math.cos(alpha) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + beta) * r
        start_y = self.y + math.cos(alpha + beta) * r
        end_x = start_x + math.sin(alpha + math.radians(45)) * s
        end_y = start_y + math.cos(alpha + math.radians(45)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + math.radians(90)) * w/2
        start_y = self.y + math.cos(alpha + math.radians(90)) * w/2
        end_x = start_x + math.sin(alpha + math.radians(90)) * s
        end_y = start_y + math.cos(alpha + math.radians(90)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + math.radians(180) - beta) * r
        start_y = self.y + math.cos(alpha + math.radians(180) - beta) * r
        end_x = start_x + math.sin(alpha + math.radians(135)) * s
        end_y = start_y + math.cos(alpha + math.radians(135)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + math.radians(180)) * h/2
        start_y = self.y + math.cos(alpha + math.radians(180)) * h/2
        end_x = start_x + math.sin(alpha + math.radians(180)) * s
        end_y = start_y + math.cos(alpha + math.radians(180)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + math.radians(180) + beta) * r
        start_y = self.y + math.cos(alpha + math.radians(180) + beta) * r
        end_x = start_x + math.sin(alpha + math.radians(225)) * s
        end_y = start_y + math.cos(alpha + math.radians(225)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha + math.radians(270)) * w / 2
        start_y = self.y + math.cos(alpha + math.radians(270)) * w / 2
        end_x = start_x + math.sin(alpha + math.radians(270)) * s
        end_y = start_y + math.cos(alpha + math.radians(270)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        start_x = self.x + math.sin(alpha - beta) * r
        start_y = self.y + math.cos(alpha - beta) * r
        end_x = start_x + math.sin(alpha + math.radians(315)) * s
        end_y = start_y + math.cos(alpha + math.radians(315)) * s
        coords.extend([(start_x, start_y, end_x, end_y)])

        if init:
            for i in range(8):
                start_x, start_y, end_x, end_y = coords[i]
                sensor = pg.shapes.Line(start_x, start_y, end_x, end_y,
                                      batch=self.batch,
                                      color=GameSettings.SENSOR_COLOR,
                                      width=GameSettings.LINE_WIDTH)
                self.sensors.append(sensor)
                self.intersection_points.append(pg.shapes.Circle(x=0, y=0, radius=GameSettings.INTERSECTION_POINT_SIZE,
                                                            batch=self.batch,
                                                            color=(200, 50, 50, 255)))
        else:
            for i in range(8):
                self.sensors[i].x, self.sensors[i].y, self.sensors[i].x2, self.sensors[i].y2 = coords[i]

    def sensor_intersections(self):
        for i in range(8):
            closest_dist = math.inf
            sensor = self.sensors[i]
            i_x_min, i_y_min = sensor.x2, sensor.y2

            # Boundaries
            for boundary in self.racetrack.boundaries:
                i_x, i_y = line_intersection([sensor.x, sensor.y], [sensor.x2, sensor.y2], [boundary.x, boundary.y],
                                             [boundary.x2, boundary.y2])
                dist = math.sqrt((i_x - sensor.x) ** 2 + (i_y - sensor.y) ** 2)
                dist_to_sensor_end = math.sqrt((i_x - sensor.x2) ** 2 + (i_y - sensor.y2) ** 2)
                if dist < closest_dist:
                    if dist_to_sensor_end < GameSettings.SENSOR_LENGTH:
                        i_x_min, i_y_min = i_x, i_y
                        closest_dist = dist
            if closest_dist < GameSettings.CAR_HIT_BOX:
                self.reset()
            else:
                self.sensor_val[i] = round(closest_dist, 1)
                self.intersection_points[i].x, self.intersection_points[i].y = i_x_min, i_y_min

        # Goals
        if self.racetrack.goals:
            if self.distance_next_goal < GameSettings.CAR_HIT_BOX:
                self.i_goals += 1
                self.i_rounds = self.i_goals // self.racetrack.n_goals
                print(f"Round {self.i_rounds}. Achieved {self.i_goals} / {self.racetrack.n_goals} goals.")
            next_goal = self.racetrack.goals[self.i_goals % self.racetrack.n_goals]
            self.distance_next_goal = round(point_to_line_distance([next_goal.x, next_goal.y], [next_goal.x2, next_goal.y2],
                                                             [self.x, self.y]),1)


    def check_collision(self):
        """1.check boundaries"""
        return False

    def check_goal(self):
        """1.check partial goal 2.check goal (time measurement, but keep going)"""
        return False

    def draw(self):
        super().draw()
        self.batch.draw()
