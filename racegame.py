import pyglet as pg
from pyglet.window import key
import math
import racetrack
from car import Car
from user_car import UserCar
from racetrack import Racetrack
from game_settings import *


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
        elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
            pass


load_status(settings.GAME_STATUS)


def update(dt):
    for obj in game_objects_to_update:
        obj.update(dt)


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/120.0)
    pg.app.run()
