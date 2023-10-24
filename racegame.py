import pyglet as pg
from pyglet.window import key
import math
import racetrack
from car import Car
from user_car import UserCar
from ai_car import AICar
from racetrack import Racetrack
from game_settings import *
from gui import GUI
from rlenv import *
from Network import Network
from utils import *
from collections import deque
import itertools
import numpy as np
import random
from torch import nn
import torch


def resize_image(img, width, height):
    img.width = width
    img.height = height
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2


def load_status(game_status):
    settings.GAME_STATUS = game_status

    game_objects.clear()
    game_objects.extend([racetrack, gui])

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    if settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.reset()
        game_objects.extend([user_car])
        gui.load_car(user_car)
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        global draw
        draw = False
        game_objects.extend([ai_car])
        gui.load_car(ai_car)


pg.resource.path = ['./resources']
pg.resource.reindex()

settings = GameSettings(game_status=GameStatus.DRAW_BOUNDARIES)
game_window = pg.window.Window(height=settings.WINDOW_HEIGHT,
                               width=settings.WINDOW_WIDTH)
draw = True

game_objects = []
game_objects_to_update = []

# Racetrack
racetrack_img = pg.resource.image('racetrack1.png')
resize_image(racetrack_img, Racetrack.IMG_WIDTH, Racetrack.IMG_HEIGHT)
racetrack = Racetrack(img=racetrack_img)

# UserCar and AICar
car_img = pg.resource.image('car.png')
resize_image(car_img, Car.IMG_WIDTH, Car.IMG_HEIGHT)
user_car = UserCar(img=car_img, racetrack=racetrack)
ai_car = AICar(img=car_img, racetrack=racetrack)

# GUI
gui = GUI(settings)

# RL Environment
rl_env = RacegameEnv(ai_car, render_mode="human")

online_net = Network(rl_env)
target_net = Network(rl_env)
target_net.load_state_dict(online_net.state_dict())  # set weights of target_net to online_net

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

obs, _ = rl_env.reset()
step = 0
episode_reward = 0.0


# Input-handlers
@game_window.event
def on_mouse_press(x, y, button, modifiers):
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        racetrack.create_boundary(x, y, button)
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        racetrack.create_goal(x, y, button)
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        pass
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        global draw
        if button == pg.window.mouse.LEFT:
            draw = not draw
        elif button == pg.window.mouse.RIGHT:
            rl_env.reset()


@game_window.event
def on_key_press(symbol, modifiers):
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.key_press(symbol, modifiers)
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        pass


@game_window.event
def on_key_release(symbol, modifiers):
    if symbol == key.M:
        next_status = math.fmod(settings.GAME_STATUS.value + 1, 4)
        load_status(GameStatus(next_status))

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.key_release(symbol, modifiers)
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        pass


@game_window.event
def on_draw():
    game_window.clear()

    if draw:
        for obj in game_objects:
            if hasattr(obj, "draw"):
                obj.draw()

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        pass
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        pass


load_status(settings.GAME_STATUS)


def rl_fill_replay_buffer():
    global obs

    action = rl_env.action_space.sample()

    new_obs, rew, done, *_ = rl_env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs, _ = rl_env.reset()


def rl_train():
    global step, obs, episode_reward
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = rl_env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, *_ = rl_env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs, _ = rl_env.reset()

        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 10:
            rl_env.reset()
            action = online_net.act(obs)
            obs, _, done, *_ = rl_env.step(action)
            if done:
                rl_env.reset()

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)  # unsqueeze(-1) to add dimensions in the end
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step & TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 500 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rew_buffer), ', Epsilon', epsilon)

        print()

    step += 1


def update(dt):
    for obj in game_objects:
        if hasattr(obj, "update_obj"):
            obj.update_obj(dt)

    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        pass
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        pass
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        pass
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        # random_action = random.randint(0, 7)
        # rl_env.step(random_action)
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            if len(replay_buffer) % 100 == 0:
                print(len(replay_buffer))
            rl_fill_replay_buffer()
        else:
            rl_train()



if __name__ == '__main__':
    pg.clock.schedule_interval(update, 1/settings.RENDER_FPS)
    pg.app.run()
