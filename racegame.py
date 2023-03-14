import pyglet as pg
import car
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
resize_image(car_img, CAR_WIDTH, CAR_HEIGHT)
center_image(car_img)


# SPRITES AND POSITIONING
racetrack = pg.sprite.Sprite(img=racetrack_img)
racetrack.x = game_window.width // 2
racetrack.y = game_window.height // 2

car = car.Car(img=car_img)
car.x = CAR_START_POSITION_X
car.y = CAR_START_POSITION_Y

# test
car.velocity_x = 100
car.velocity_y = 100
car.rotation_speed = 30
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
