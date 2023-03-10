import pyglet as pg
import car
from game_constants import *

pg.resource.path = ['./resources']
pg.resource.reindex()

game_window = pg.window.Window(height=WINDOW_HEIGHT, width=WINDOW_WIDTH)


def center_image(img):
    img.anchor_x = img.width//2
    img.anchor_y = img.height//2


def center_sprite(sprite):
    sprite.x = game_window.width // 2
    sprite.y = game_window.height // 2


# IMAGES
racetrack_img = pg.resource.image('racetrack.png')
racetrack_img.width = RACETRACK_WIDTH
racetrack_img.height = RACETRACK_HEIGHT
center_image(racetrack_img)

car_img = pg.resource.image('car.png')
car_img.width = CAR_WIDTH
car_img.height = CAR_HEIGHT
center_image(car_img)


# SPRITES
racetrack = pg.sprite.Sprite(img=racetrack_img)
center_sprite(racetrack)

car = car.Car(img=car_img)
car.x = CAR_START_POSITION_X
car.y = CAR_START_POSITION_Y

# test
car.velocity_x = 100
car.velocity_y = 100
car.velocity_rotation = 30
# test end


def update(dt):
    car.update(dt)


@game_window.event
def on_draw():
    game_window.clear()
    racetrack.draw()
    car.draw()


if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/120.0)
    pg.app.run()
