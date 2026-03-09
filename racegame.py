import os
# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pyglet as pg
from pyglet.window import key
import math
from game.car import Car
from game.user_car import UserCar
from game.ai_car import AICar
from game.racetrack import Racetrack
from game.game_settings import *
from game.gui import GUI
from game.utils import *
from training.rlenv import *
from training.network import Network
from training.training_monitor import TrainingMonitor, save_model
from training.training_config import *
from collections import deque
import itertools
import numpy as np
import random
import torch
from torch import nn
import time
import glob

# ========================
# CHECKPOINT FUNCTIONALITY
# ========================
def find_latest_checkpoint():
    """Find the latest checkpoint automatically."""
    model_files = glob.glob("models/model_step_*.pth")
    if not model_files:
        return None, 0
    
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    step_number = int(latest_model.split('_')[-1].split('.')[0])
    return latest_model, step_number

def load_checkpoint_if_available(online_net, target_net):
    """Load checkpoint if auto-resume is enabled."""
    if not AUTO_RESUME:
        return False, 0, EPSILON_START, deque(maxlen=100)
    
    model_path, step_number = find_latest_checkpoint()
    if not model_path:
        print("No checkpoint found - starting fresh training")
        return False, 0, EPSILON_START, deque(maxlen=100)
    
    try:
        print(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Load model weights
        online_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.eval()
        
        # Restore training state
        step = step_number
        epsilon = checkpoint.get('epsilon', EPSILON_END)
        avg_reward = checkpoint.get('avg_reward', 0.0)
        
        # Initialize reward buffer with checkpoint value
        rew_buffer = deque([avg_reward], maxlen=100)
        
        print(f"Resumed from step {step:,}")
        print(f"Epsilon: {epsilon:.4f}")
        print(f"Avg Reward: {avg_reward:.2f}")
        
        return True, step, epsilon, rew_buffer
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training instead")
        return False, 0, EPSILON_START, deque(maxlen=100)


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
target_net.eval()  # Target net must stay in eval mode - dropout would make targets stochastic

# Optimizer with config learning rate
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# Initialize training variables with checkpoint loading
checkpoint_loaded, step, epsilon, rew_buffer = load_checkpoint_if_available(online_net, target_net)

if not checkpoint_loaded:
    # Fresh training initialization
    target_net.load_state_dict(online_net.state_dict())
    step = 0
    epsilon = EPSILON_START
    rew_buffer = deque([0.0], maxlen=100)
    print("STARTING FRESH TRAINING")
else:
    print(f"RESUMING TRAINING from step {step:,}")

obs, _ = rl_env.reset()
episode_reward = 0.0
env_step = 0  # Counts real environment steps (one per game frame) - used for epsilon decay


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


def rl_env_step():
    """Take one action in the environment and store the transition. Called once per game frame."""
    global obs, episode_reward, env_step

    epsilon = np.interp(env_step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if random.random() <= epsilon:
        action = rl_env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, *_ = rl_env.step(action)
    replay_buffer.append((obs, action, rew, done, new_obs))
    obs = new_obs
    episode_reward += rew
    env_step += 1

    if done:
        obs, _ = rl_env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0


def rl_gradient_step():
    """Sample a batch and do one gradient update. Called GRADIENT_STEPS_PER_FRAME times per frame."""
    global step

    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses_t     = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32, device=DEVICE)
    actions_t   = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64,   device=DEVICE).unsqueeze(-1)
    rews_t      = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    dones_t     = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    new_obses_t = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        max_target_q = target_net(new_obses_t).max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q

    action_q_values = torch.gather(online_net(obses_t), dim=1, index=actions_t)
    loss = nn.functional.mse_loss(action_q_values, targets)

    if torch.isnan(loss):
        print("WARNING: Loss is NaN, skipping gradient step")
        return

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

    epsilon = np.interp(env_step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    avg_reward_100 = np.mean(rew_buffer) if len(rew_buffer) > 0 else 0
    last_episode_reward = rew_buffer[-1] if len(rew_buffer) > 0 else episode_reward
    monitor.log_step(step, last_episode_reward, loss.item(), epsilon, avg_reward_100)

    if step % LOG_FREQ == 0:
        print()
        print('=' * 50)
        print(f'Gradient step: {step}  |  Env step: {env_step}')
        print(f'Avg Reward (last 100): {avg_reward_100:.2f}')
        print(f'Current Episode Reward: {episode_reward:.2f}')
        print(f'Last Completed Episode: {last_episode_reward:.2f}')
        print(f'Epsilon: {epsilon:.3f}')
        print(f'Loss: {loss.item():.4f}')
        print(f'Replay Buffer Size: {len(replay_buffer)}')
        print(f'Device: {DEVICE}')
        if len(rew_buffer) > 0:
            print(f'Max Reward: {max(rew_buffer):.2f}')
            print(f'Min Reward: {min(rew_buffer):.2f}')
        print('=' * 50)

        if step % SAVE_FREQ == 0:
            model_path = f"models/model_step_{step}.pth"
            os.makedirs("models", exist_ok=True)
            save_model(online_net, model_path, {
                'step': step,
                'env_step': env_step,
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

    if step % PLOT_FREQ == 0 and step > 0:
        print(f"Generating plot at step {step}...")
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
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            if len(replay_buffer) % 500 == 0:
                print(f"Filling replay buffer: {len(replay_buffer)}/{MIN_REPLAY_SIZE}")
            rl_fill_replay_buffer()
        else:
            if env_step % 100 == 0:
                print(f"Env step: {env_step} | Gradient step: {step} | Buffer: {len(replay_buffer)}")
            rl_env_step()                          # one real environment step per game frame
            for _ in range(GRADIENT_STEPS_PER_FRAME):
                rl_gradient_step()                 # N gradient updates on the replay buffer


pg.clock.schedule_interval(update, 1/settings.RENDER_FPS)

if __name__ == '__main__':
    pg.app.run()
