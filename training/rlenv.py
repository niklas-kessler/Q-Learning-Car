import gymnasium as gym
from gymnasium import spaces
import math
from game.game_settings import *
from .training_config import *  # Import ALL hyperparameters from central config


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

    def _get_goal_direction(self):
        """
        Returns (angle_norm, distance_norm) toward the next goal midpoint, both relative to the car.
          angle_norm:    [-1, 1]  — 0 = straight ahead, ±1 = directly behind
          distance_norm: [0, 1]  — 0 = at goal, 1 = maximum possible distance (window diagonal)
        Returns (0.0, 1.0) when no goals are defined.
        """
        if not self.car.racetrack.goals:
            return 0.0, 1.0

        next_goal = self.car.racetrack.goals[self.car.i_goals % self.car.racetrack.n_goals]
        goal_mid_x = (next_goal.x + next_goal.x2) / 2
        goal_mid_y = (next_goal.y + next_goal.y2) / 2

        dx = goal_mid_x - self.car.x
        dy = goal_mid_y - self.car.y

        # Distance normalized by window diagonal
        dist = math.sqrt(dx ** 2 + dy ** 2)
        max_dist = math.sqrt(GameSettings.WINDOW_WIDTH ** 2 + GameSettings.WINDOW_HEIGHT ** 2)
        distance_norm = min(1.0, dist / max_dist)

        # Angle relative to car heading, normalized to [-1, 1]
        # Pyglet: rotation=0 faces +y (up), increases clockwise → heading vector = (sin(r), cos(r))
        world_angle = math.atan2(dx, dy)  # angle from +y axis, clockwise
        car_heading = math.radians(self.car.rotation)
        relative_angle = world_angle - car_heading
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi  # clamp to [-π, π]
        angle_norm = relative_angle / math.pi

        return angle_norm, distance_norm

    def _get_obs(self):
        """
        Returns a list of 11 floats:
          [0-7] Sensor distances normalized to [0, 1]  (0 = wall touching, 1 = max sensor range)
                order: f, fr, r, br, b, bl, l, fl
          [8]   Car velocity normalized to [-1, 1]     (-1 = full reverse, 0 = stopped, 1 = full forward)
          [9]   Angle to next goal normalized to [-1, 1] (0 = straight ahead, ±1 = behind)
          [10]  Distance to next goal normalized to [0, 1] (0 = at goal, 1 = max distance)
        """
        # Normalize sensor values to [0, 1]
        sensor_vals = []
        for val in self.car.sensor_val:
            if math.isnan(val) or math.isinf(val):
                sensor_vals.append(1.0)
            else:
                sensor_vals.append(max(0.0, min(float(val), float(GameSettings.SENSOR_LENGTH))) / GameSettings.SENSOR_LENGTH)

        # Normalize velocity to [-1, 1]
        from game.car import Car
        velocity = self.car.velocity
        if math.isnan(velocity) or math.isinf(velocity):
            velocity = 0.0
        velocity_norm = max(-1.0, min(1.0, velocity / Car.MAX_VELOCITY))

        angle_norm, distance_norm = self._get_goal_direction()

        return sensor_vals + [velocity_norm, angle_norm, distance_norm]

    def _get_info(self):
        return {
            "Reward": getattr(self.car, 'reward', 0.0)
        }

    def step(self, action):
        self.car.action(action)

        terminated = self.car.collision
        goal = self.car.goal

        if terminated:
            reward = CRASH_PENALTY
        elif goal:
            reward = GOAL_REWARD
        else:
            reward = SURVIVAL_REWARD

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
