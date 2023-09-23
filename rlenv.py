import gymnasium as gym
from gymnasium import spaces
import numpy as np

import pyglet as pg
from pyglet.window import key
import math
import racetrack
from car import Car
from user_car import UserCar
from racetrack import Racetrack
from game_settings import *
from utils import *


class RacegameEnv(gym.Env):
    metadata = {"render_modes": ["human"],  "render_fps": 4}

    pg.resource.path = ['./resources']
    pg.resource.reindex()

    event_stack_size = 0

    game_objects = []
    game_objects_to_update = []

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict(
            {
                'f': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'fr': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'r': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'br': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'b': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'bl': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'l': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
                'fl': spaces.Box(0, GameSettings.SENSOR_LENGTH, shape=(1,), dtype=float),
            }
        )

        # 0=f, 1=fr, ...
        # maybe instead model with Sequence(4)-space, allowing variable number of actions
        self.action_space = spaces.Discrete(8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        settings = GameSettings(game_status=GameStatus.DRAW_BOUNDARIES)
        game_window = None

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def render(self):
        pass

    def _render_frame(self):
        # game_window = pg.window.Window(height=settings.WINDOW_HEIGHT,
        #                                width=settings.WINDOW_WIDTH)
        pass

    def close(self):
        pass
