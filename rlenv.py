import gymnasium as gym
from gymnasium import spaces

from game_settings import *


GAMMA = 0.99  # discount rate traget learning
BATCH_SIZE = 32  # how many transistions to sample from buffer
BUFFER_SIZE = 50000  # how many to store before overwriting all transitions
MIN_REPLAY_SIZE = 3000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 20000
TARGET_UPDATE_FREQ = 1000


class RacegameEnv(gym.Env):
    metadata = {"render_modes": ["human"],  "render_fps": GameSettings.RENDER_FPS}

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

        self.car = ai_car

    def _get_obs(self):
        """
        return {'f': self.car.sensor_val[0],
                'fr': self.car.sensor_val[1],
                'l': self.car.sensor_val[2],
                'r': self.car.sensor_val[3],
                'bl': self.car.sensor_val[4],
                'b': self.car.sensor_val[5],
                'br': self.car.sensor_val[6],
                'fl': self.car.sensor_val[7]}
        """
        return self.car.sensor_val

    def _get_info(self):
        return {
            "Reward": self.car.reward
        }

    def step(self, action):
        self.car.action(action)

        # TODO
        terminated = self.car.collision
        goal = self.car.goal
        if terminated:
            reward = -15
        elif goal:
            reward = 1
        else:
            reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def _render_frame(self):
        pass
    #     if self.game_window is None and self.render_mode == "human":
    #         game_window = pg.window.Window(height=self.settings.WINDOW_HEIGHT,
    #                                        width=self.settings.WINDOW_WIDTH)

    # def render(self):
    #      pass

    def reset(self, seed=None, options=None):
        self.car.reset()
        return self._get_obs(), self._get_info()

    # def close(self):
    #     pass
