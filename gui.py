import pyglet as pg
from car import Car
from game_settings import GameSettings


class GUI:
    def __init__(self, settings: GameSettings):
        self.car = None
        self.settings = settings
        self.gui_batch = pg.graphics.Batch()

        self.game_status_label = pg.text.Label(text="Game status: ",
                                               x=GameSettings.GAME_STATUS_LABEL_X,
                                               y=GameSettings.GAME_STATUS_LABEL_Y,
                                               font_size=GameSettings.FONT_SIZE,
                                               batch=self.gui_batch)
        self.game_status_value_label = pg.text.Label(text="init",
                                               x=GameSettings.GAME_STATUS_LABEL_X + 90,
                                               y=GameSettings.GAME_STATUS_LABEL_Y,
                                               font_size=GameSettings.FONT_SIZE,
                                               batch=self.gui_batch)
        self.dist_labels = []
        self.dist_value_labels = []
        s = ['f', 'fr', 'r', 'br', 'b', 'bl', 'l', 'fl']
        for i in range(8):
            self.dist_labels.append(pg.text.Label(text=s[i] + ': ',
                                                  x=GameSettings.DIST_LABELS_POSITION_X,
                                                  y=GameSettings.DIST_LABELS_POSITION_Y - i * 15,
                                                  font_size=GameSettings.FONT_SIZE,
                                                  batch=self.gui_batch))
            self.dist_value_labels.append(pg.text.Label(text="init",
                                                        x=GameSettings.DIST_LABELS_POSITION_X + 30,
                                                        y=GameSettings.DIST_LABELS_POSITION_Y - i * 15,
                                                        font_size=GameSettings.FONT_SIZE,
                                                        batch=self.gui_batch))

        self.i_goal_label = pg.text.Label(text="Goals achieved: ",
                                          x=GameSettings.GOAL_LABEL_POSITION_X,
                                          y=GameSettings.GOAL_LABEL_POSITION_Y,
                                          font_size=GameSettings.FONT_SIZE,
                                          batch=self.gui_batch)
        self.i_goal_value_label = pg.text.Label(text="init",
                                                x=GameSettings.GOAL_LABEL_POSITION_X+150,
                                                y=GameSettings.GOAL_LABEL_POSITION_Y,
                                                font_size=GameSettings.FONT_SIZE,
                                                batch=self.gui_batch)
        self.distance_goal_label = pg.text.Label(text="Distance to next goal: ",
                                                 x=GameSettings.GOAL_LABEL_POSITION_X,
                                                 y=GameSettings.GOAL_LABEL_POSITION_Y-15,
                                                 font_size=GameSettings.FONT_SIZE,
                                                 batch=self.gui_batch)
        self.distance_goal_value_label = pg.text.Label(text="init",
                                                       x=GameSettings.GOAL_LABEL_POSITION_X + 150,
                                                       y=GameSettings.GOAL_LABEL_POSITION_Y-15,
                                                       font_size=GameSettings.FONT_SIZE,
                                                       batch=self.gui_batch)

    def load_car(self, car: Car):
        self.car = car

    def update_obj(self, dt):
        if self.settings is not None:
            self.game_status_value_label.text = str(self.settings.GAME_STATUS)
        if self.car is not None:
            for i in range(8):
                self.dist_value_labels[i].text = str(self.car.sensor_val[i])
            self.i_goal_value_label.text = str(self.car.i_goals)
            self.distance_goal_value_label.text = str(self.car.distance_next_goal)

    def draw(self):
        self.gui_batch.draw()
