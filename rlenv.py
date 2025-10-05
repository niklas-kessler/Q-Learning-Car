import gymnasium as gym
from gymnasium import spaces
from game_settings import *
from training_config import *  # Import ALL hyperparameters from central config


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
        # Store previous distance for reward calculation
        prev_distance = self.car.distance_next_goal
        prev_velocity = self.car.velocity
        
        self.car.action(action)

        terminated = self.car.collision
        goal = self.car.goal
        
        # Improved reward structure using config parameters
        reward = 0.0
        
        if terminated:
            reward = CRASH_PENALTY  # Heavy penalty for crashing
        elif goal:
            reward = GOAL_REWARD   # Big reward for reaching goal
        else:
            # Distance-based reward (encourage getting closer to goal)
            current_distance = self.car.distance_next_goal
            if current_distance < prev_distance:
                reward += DISTANCE_REWARD_SCALE  # Reward for getting closer
            else:
                reward -= DISTANCE_REWARD_SCALE * 0.2  # Small penalty for getting further
            
            # Velocity-based reward (encourage movement but not too fast)
            speed = abs(self.car.velocity)
            if speed > 50:  # Encourage some speed
                reward += VELOCITY_REWARD_SCALE
            elif speed < 10:  # Discourage standing still
                reward -= VELOCITY_REWARD_SCALE * 2
                
            # Sensor-based reward (avoid walls)
            min_sensor_val = min(self.car.sensor_val)
            if min_sensor_val < 20:  # Close to wall
                reward -= SENSOR_PENALTY_SCALE
            elif min_sensor_val > 50:  # Safe distance from walls
                reward += SENSOR_PENALTY_SCALE * 0.3
                
            # Small positive reward for surviving
            reward += SURVIVAL_REWARD

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
