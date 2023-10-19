import pyglet as pg
from car import Car
from game_settings import GameSettings


class GUI:
    def __init__(self):
        self.gui_batch = pg.graphics.Batch()
        self.dist_labels = []
        self.dist_value_labels = []
        self.car = None

        ax = GameSettings.WINDOW_WIDTH - 150
        ay = GameSettings.WINDOW_HEIGHT - 50
        s = ['f', 'fr', 'r', 'br', 'b', 'bl', 'l', 'fl']

        for i in range(8):
            self.dist_labels.append(pg.text.Label(text=s[i] + ': ', x=ax, y=ay - i * 15, font_size=10,
                                                  batch=self.gui_batch))
            self.dist_value_labels.append(pg.text.Label(text="init", x=ax + 30, y=ay - i * 15,
                                                        font_size=10, batch=self.gui_batch))

    def load_car(self, car: Car):
        self.car = car

    def update_obj(self, dt):
        if self.car is not None:
            for i in range(8):
                self.dist_value_labels[i].text = str(self.car.sensor_val[i])

    def draw(self):
        self.gui_batch.draw()
