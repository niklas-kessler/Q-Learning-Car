import pyglet as pg
from car import Car
from game_constants import *


def center_image(img):
    img.anchor_x = img.width//2
    img.anchor_y = img.height//2


def resize_image(img, width, height):
    img.width = width
    img.height = height


pg.resource.path = ['./resources']
pg.resource.reindex()

game_window = pg.window.Window(height=WINDOW_HEIGHT, width=WINDOW_WIDTH)


# IMAGES
racetrack_img = pg.resource.image('racetrack.png')
resize_image(racetrack_img, RACETRACK_WIDTH, RACETRACK_HEIGHT)
center_image(racetrack_img)

car_img = pg.resource.image('car.png')
resize_image(car_img, Car.IMG_WIDTH, Car.IMG_HEIGHT)
center_image(car_img)


# SPRITES AND POSITIONING
racetrack = pg.sprite.Sprite(img=racetrack_img)
racetrack.x = game_window.width // 2
racetrack.y = game_window.height // 2

agent = Car(img=car_img)

game_window.push_handlers(agent)


def update(dt):
    agent.update(dt)


@game_window.event
def on_draw():
    game_window.clear()
    racetrack.draw()
    agent.draw()


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/120.0)
    pg.app.run()
