import pyglet as pg
from pyglet.window import key
import math
import racetrack
from car import Car
from user_car import UserCar
from racetrack import Racetrack
from game_settings import *
from utils import *


def resize_image(img, width, height):
    img.width = width
    img.height = height
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2


def load_status(game_status):
    global event_stack_size

    settings.GAME_STATUS = game_status

    game_objects.clear()
    game_objects_to_update.clear()

    while event_stack_size > 0:
        game_window.pop_handlers()
        event_stack_size -= 1

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        game_objects.extend([racetrack])
        game_window.push_handlers(racetrack)
        event_stack_size += 1

    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.reset()
        game_objects.extend([racetrack, user_car])
        game_objects_to_update.extend([user_car])
        game_window.push_handlers(user_car)
        event_stack_size += 1

    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        game_objects.extend([racetrack])


def load_gui():
    global dist_labels, dist_value_labels
    ax = settings.WINDOW_WIDTH - 150
    ay = settings.WINDOW_HEIGHT - 50
    s = ['f', 'fr', 'r', 'br', 'b', 'bl', 'l', 'fl']

    for i in range(8):
        dist_labels.append(pg.text.Label(text=s[i]+': ', x=ax, y=ay-i*15, font_size=10,
                                         batch=gui_batch))
        dist_value_labels.append(pg.text.Label(text="init", x=ax+30, y=ay-i*15,
                                               font_size=10, batch=gui_batch))
        intersection_points.append(pg.shapes.Circle(x=0, y=0, radius=4,
                                                    batch=gui_batch,
                                                    color=(200, 50, 50, 255)))


pg.resource.path = ['./resources']
pg.resource.reindex()

settings = GameSettings(game_status=GameStatus.DRAW_BOUNDARIES)
game_window = pg.window.Window(height=settings.WINDOW_HEIGHT,
                               width=settings.WINDOW_WIDTH)

event_stack_size = 0

game_objects = []
game_objects_to_update = []

# IMAGES
racetrack_img = pg.resource.image('racetrack1.png')
resize_image(racetrack_img, Racetrack.IMG_WIDTH, Racetrack.IMG_HEIGHT)

car_img = pg.resource.image('car.png')
resize_image(car_img, Car.IMG_WIDTH, Car.IMG_HEIGHT)

# SPRITES AND POSITIONING
racetrack = Racetrack(img=racetrack_img)
user_car = UserCar(img=car_img)

# gui
gui_batch = pg.graphics.Batch()
dist_labels = []
dist_value_labels = []
intersection_points = []
load_gui()


@game_window.event
def on_key_release(symbol, modifiers):
    if symbol == key.M:
        next_status = math.fmod(settings.GAME_STATUS.value + 1, 3)
        load_status(GameStatus(next_status))
        print(settings.GAME_STATUS)


@game_window.event
def on_draw():
    game_window.clear()
    for obj in game_objects:
        obj.draw()
    racetrack.racetrack_batch.draw()

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    else:
        if settings.GAME_STATUS == GameStatus.USER_CONTROLS:
            user_car.car_batch.draw()
            gui_batch.draw()
        elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
            pass


load_status(settings.GAME_STATUS)


def sensor_intersections(car: Car):
    for i in range(8):
        closest_dist = math.inf
        sensor = car.sensors[i]
        i_x_min, i_y_min = sensor.x2, sensor.y2
        for boundary in racetrack.boundaries:
            i_x, i_y = line_intersection([sensor.x, sensor.y], [sensor.x2, sensor.y2], [boundary.x, boundary.y],
                                                 [boundary.x2, boundary.y2])
            dist = math.sqrt((i_x - sensor.x)**2 + (i_y - sensor.y)**2)
            dist_to_sensor_end = math.sqrt((i_x - sensor.x2)**2 + (i_y - sensor.y2)**2)
            if dist < closest_dist:
                if dist_to_sensor_end < settings.SENSOR_LENGTH:
                    i_x_min, i_y_min = i_x, i_y
                    closest_dist = dist
        if closest_dist < settings.CAR_HIT_BOX:
            car.game_over()
        else:
            dist_value_labels[i].text = str(round(closest_dist, 1))
            intersection_points[i].x, intersection_points[i].y = i_x_min, i_y_min


def update(dt):
    for obj in game_objects_to_update:
        obj.update(dt)
    sensor_intersections(user_car)


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/120.0)
    pg.app.run()
