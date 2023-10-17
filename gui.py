import pyglet as pg
from game_settings import GameSettings

class GUI:
    def __init__(self):
        self.gui_batch = pg.graphics.Batch()
        self.dist_labels = []
        self.dist_value_labels = []

    def load(self):
        ax = GameSettings.WINDOW_WIDTH - 150
        ay = GameSettings.WINDOW_HEIGHT - 50
        s = ['f', 'fr', 'r', 'br', 'b', 'bl', 'l', 'fl']

        for i in range(8):
            self.dist_labels.append(pg.text.Label(text=s[i] + ': ', x=ax, y=ay - i * 15, font_size=10,
                                             batch=self.gui_batch))
            self.dist_value_labels.append(pg.text.Label(text="init", x=ax + 30, y=ay - i * 15,
                                                   font_size=10, batch=self.gui_batch))
