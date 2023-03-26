import pyglet as pg

import racetrack
from car import Car
from user_car import UserCar
from racetrack import Racetrack
from game_settings import *


# TODO: REPLACE with actual cmd, sth like if game_window.handlers > 0
handlers_pushed = False


def resize_image(img, width, height):
    img.width = width
    img.height = height
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2


def load_status(game_status):
    settings.GAME_STATUS = game_status

    game_objects.clear()
    game_objects_to_update.clear()

    if handlers_pushed:
        game_window.pop_handlers()

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        game_objects.extend([racetrack])
        game_window.push_handlers(racetrack)

    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        game_objects.extend([racetrack, user_car])
        game_objects_to_update.extend([user_car])
        game_window.push_handlers(user_car)

    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        game_objects.extend([racetrack])


pg.resource.path = ['./resources']
pg.resource.reindex()

settings = GameSettings(game_status=GameStatus.DRAW_BOUNDARIES)
game_window = pg.window.Window(height=settings.WINDOW_HEIGHT,
                               width=settings.WINDOW_WIDTH)

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


load_status(settings.GAME_STATUS)
handlers_pushed = True


def update(dt):
    for obj in game_objects_to_update:
        obj.update(dt)
    """
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        print("draw boundaries")
    else:
        if settings.GAME_STATUS == GameStatus.USER_CONTROLS:
            print("user controls")
        elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
            print("ai train")
    """


@game_window.event
def on_draw():
    game_window.clear()
    for obj in game_objects:
        obj.draw()
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        racetrack.racetrack_batch.draw()
    else:
        if settings.GAME_STATUS == GameStatus.USER_CONTROLS:
            pass
        elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
            pass


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/120.0)
    pg.app.run()
