import pyglet as pg

import racetrack
import user_controls
from car import Car
from racetrack import Racetrack
from game_settings import *


# TODO: REPLACE with actual cmd, sth like if game_window.handlers > 0
handlers_pushed = False


def resize_image(img, width, height):
    img.width = width
    img.height = height
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2


def load(game_status):
    settings.GAME_STATUS = game_status

    game_objects.clear()

    if handlers_pushed:
        game_window.pop_handlers()

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        game_objects.extend([racetrack])
        game_window.push_handlers(racetrack)

    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        game_objects.append([racetrack, user_car])
        game_window.push_handlers(user_controls.on_key_press(user_car),
                                  user_controls.on_key_release(user_car))

    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        game_objects.extend([racetrack])


pg.resource.path = ['./resources']
pg.resource.reindex()

settings = GameSettings()
game_window = pg.window.Window(height=settings.WINDOW_HEIGHT,
                               width=settings.WINDOW_WIDTH)

game_objects = []

# IMAGES
racetrack_img = pg.resource.image('racetrack1.png')
resize_image(racetrack_img, Racetrack.IMG_WIDTH, Racetrack.IMG_HEIGHT)

car_img = pg.resource.image('car.png')
resize_image(car_img, Car.IMG_WIDTH, Car.IMG_HEIGHT)

# SPRITES AND POSITIONING
racetrack = Racetrack(img=racetrack_img)
user_car = Car(img=car_img, user_controls=True)


load(GameStatus.DRAW_BOUNDARIES)
handlers_pushed = True


def update(dt):
    for obj in game_objects:
        obj.update
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
