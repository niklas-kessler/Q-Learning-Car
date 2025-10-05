import os
# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
import torch
from torch import nn
from training_monitor import TrainingMonitor, save_model
from training_config import *


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
        ai_car.reset()  # Reset AI car position
        game_objects.extend([ai_car])
        gui.load_car(ai_car)
        print("AI Training mode activated - car should be visible")


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

# Training monitoring
monitor = TrainingMonitor()

# GUI
gui = GUI(settings)

# RL Environment
rl_env = RacegameEnv(ai_car, render_mode="human")

# Neural Networks - using central config for device
online_net = Network(rl_env)
target_net = Network(rl_env)
target_net.load_state_dict(online_net.state_dict())

# Optimizer with config learning rate
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
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
        # Right click resets the environment during training
        if button == pg.window.mouse.RIGHT:
            rl_env.reset()
            print("Environment reset during training")


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
    # SPACE key to switch between modes
    if symbol == key.SPACE:
        next_status = math.fmod(settings.GAME_STATUS.value + 1, 4)
        load_status(GameStatus(next_status))
        print(f"Switched to: {settings.GAME_STATUS}")

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
        gui.draw()
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        gui.draw()
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        gui.draw()
    elif settings.GAME_STATUS == GameStatus.AI_TRAIN:
        gui.draw()


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


def save_training_results():
    """Save final training results and generate summary plots."""
    print("\nSaving training results...")
    
    try:
        # Save final model
        os.makedirs("models", exist_ok=True)
        final_model_path = f"models/final_model_{monitor.session_id}.pth"
        save_model(online_net, final_model_path, {
            'final_step': step,
            'final_avg_reward': np.mean(rew_buffer) if len(rew_buffer) > 0 else 0,
            'total_episodes': len(monitor.episode_rewards),
            'session_id': monitor.session_id
        })
        
        # Save metrics and generate final plots
        monitor.save_metrics()
        monitor.plot_training_progress()
        monitor.print_summary()
        
        print(f"Training session {monitor.session_id} results saved!")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        print("Training data may not have been saved.")


def rl_train():
    global step, obs, episode_reward
    
    # Only train if we have enough samples
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return
    
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    # Choose action with epsilon-greedy policy
    if random.random() <= epsilon:
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

    # Automatic continuation if performing well
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 200:  # Adjusted threshold for new reward system
            obs, _ = rl_env.reset()
            action = online_net.act(obs)
            obs, _, done, *_ = rl_env.step(action)
            if done:
                obs, _ = rl_env.reset()

    # Start Gradient Step - sample batch for training
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    # Convert to tensors and move to device from config
    obses_t = torch.as_tensor(obses, dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=DEVICE)

    # Compute targets using target network
    with torch.no_grad():
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute current Q values
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    
    # Compute loss with gradient clipping
    loss = nn.functional.mse_loss(action_q_values, targets)  # Changed to MSE loss

    # Gradient Descent Step with clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)  # Gradient clipping
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Enhanced Logging and Monitoring using config frequencies
    if step % LOG_FREQ == 0:
        avg_reward_100 = np.mean(rew_buffer) if len(rew_buffer) > 0 else 0
        
        # Log to monitor
        monitor.log_step(step, episode_reward, loss.item(), epsilon, avg_reward_100)
        
        print()
        print('=' * 50)
        print(f'Step: {step}')
        print(f'Avg Reward (last 100): {avg_reward_100:.2f}')
        print(f'Epsilon: {epsilon:.3f}')
        print(f'Loss: {loss.item():.4f}')
        print(f'Replay Buffer Size: {len(replay_buffer)}')
        print(f'Device: {DEVICE}')
        if len(rew_buffer) > 0:
            print(f'Max Reward: {max(rew_buffer):.2f}')
            print(f'Min Reward: {min(rew_buffer):.2f}')
        print('=' * 50)
        
        # Save model and metrics periodically
        if step % SAVE_FREQ == 0:
            model_path = f"models/model_step_{step}.pth"
            os.makedirs("models", exist_ok=True)
            save_model(online_net, model_path, {
                'step': step,
                'avg_reward': avg_reward_100,
                'epsilon': epsilon,
                'hyperparameters': {
                    'learning_rate': LEARNING_RATE,
                    'batch_size': BATCH_SIZE,
                    'buffer_size': BUFFER_SIZE,
                    'gamma': GAMMA
                }
            })
            monitor.save_metrics()
            
        # Generate plots using config frequency
        if step % PLOT_FREQ == 0 and step > 0:
            monitor.plot_training_progress()

    step += 1


def update(dt):
    #print(f"Update called with dt={dt:.4f}s")  # Debug log for update calls
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
        # Training logic
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            if len(replay_buffer) % 500 == 0:  # Log every 500 instead of 100 for less spam
                print(f"Filling replay buffer: {len(replay_buffer)}/{MIN_REPLAY_SIZE}")
            rl_fill_replay_buffer()
        else:
            if step % 100 == 0:  # Quick debug log every 100 steps
                print(f"Training step: {step}")
            rl_train()


pg.clock.schedule_interval(update, 1/settings.RENDER_FPS)

if __name__ == '__main__':
    pg.app.run()
