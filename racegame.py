import pyglet as pg
from pyglet.window import key
import math
import racetrack
from car import Car
from user_car import UserCar
from ai_car import AICar
from racetrack import Racetrack
from game_settings import *
from gui import GUI
from rlenv import RacegameEnv
from Network import Network
from utils import *
import random


def resize_image(img, width, height):
    img.width = width
    img.height = height
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2


def load_status(game_status):
    global event_stack_size

    settings.GAME_STATUS = game_status

    game_objects.clear()
    game_objects.extend([racetrack, gui])

    while event_stack_size > 0:
        game_window.pop_handlers()
        event_stack_size -= 1

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        game_window.push_handlers(racetrack)
        event_stack_size += 1

    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.reset()
        game_objects.extend([user_car])
        game_window.push_handlers(user_car)
        event_stack_size += 1
        gui.load_car(user_car)

    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        ai_car.reset()
        game_objects.extend([ai_car])
        gui.load_car(ai_car)


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
user_car = UserCar(img=car_img, racetrack=racetrack)
ai_car = AICar(img=car_img, racetrack=racetrack)

# GUI
gui = GUI()

# RL_ENV
rl_env = RacegameEnv(ai_car, render_mode="human")
online_net = Network(rl_env)
target_net = Network(rl_env)

@game_window.event
def on_key_release(symbol, modifiers):
    if symbol == key.M:
        next_status = math.fmod(settings.GAME_STATUS.value + 1, 3)
        load_status(GameStatus(next_status))


@game_window.event
def on_draw():
    game_window.clear()

    for obj in game_objects:
        if hasattr(obj, "draw"):
            obj.draw()

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        pass
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        pass


load_status(settings.GAME_STATUS)


def update(dt):
    for obj in game_objects:
        if hasattr(obj, "update_obj"):
            obj.update_obj(dt)

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        pass
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        random_action = random.randint(0, 7)
        rl_env.step(random_action)


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/settings.RENDER_FPS)
    pg.app.run()
