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
    metadata = {"render_modes": ["human"],  "render_fps": GameSettings.RENDER_FPS}

    pg.resource.path = ['./resources']
    pg.resource.reindex()

    event_stack_size = 0

    game_objects = []
    game_objects_to_update = []

    def __init__(self, ai_car, render_mode=None):
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

        # TODO: maybe instead model with Sequence(4)-space, allowing variable number of actions
        # 0=fl, 1=f, ...
        self.action_space = spaces.Discrete(8)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.settings = GameSettings(game_status=GameStatus.DRAW_BOUNDARIES)
        self.game_window = None

        self.car = ai_car

    def _get_obs(self):
        return self.car.sensor_val

    def _get_info(self):
        return {
            "Reward": self.car.reward
        }

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        self.car.action(action)

        # TODO
        terminated = self.car.check_collision()
        goal = self.car.check_goal()
        reward = 1 if goal else 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
         pass

    def _render_frame(self):
        pass
    #     if self.game_window is None and self.render_mode == "human":
    #         game_window = pg.window.Window(height=self.settings.WINDOW_HEIGHT,
    #                                        width=self.settings.WINDOW_WIDTH)

    def close(self):
        pass

