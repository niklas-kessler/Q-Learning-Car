#!/usr/bin/env python3
"""
Resume Training Script - Continue training from the latest checkpoint.
"""
import os
# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import glob
import json
from training_config import *

def find_latest_checkpoint():
    """Find the latest model checkpoint and training state."""
    print("🔍 Searching for checkpoints...")
    
    # Find all model files
    model_files = glob.glob("models/model_step_*.pth")
    final_models = glob.glob("models/final_model_*.pth")
    
    if not model_files and not final_models:
        print("❌ No checkpoints found! Start fresh training instead.")
        return None, None, None
    
    # Prefer step models over final models for resuming
    if model_files:
        latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        step_number = int(latest_model.split('_')[-1].split('.')[0])
        model_type = "checkpoint"
    else:
        latest_model = max(final_models, key=lambda x: x.split('_')[-1].split('.')[0])
        # Extract step from final model metadata
        try:
            checkpoint = torch.load(latest_model, map_location='cpu')
            step_number = checkpoint.get('step', 0)
            model_type = "final"
        except:
            print("❌ Could not load final model metadata.")
            return None, None, None
    
    # Find corresponding training log
    training_logs = glob.glob("training_logs/training_metrics_*.json")
    matching_log = None
    
    if training_logs:
        # Try to find log with matching timestamp or closest step count
        for log_file in training_logs:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    if abs(log_data.get('total_steps', 0) - step_number) < 10000:
                        matching_log = log_file
                        break
            except:
                continue
        
        if not matching_log:
            matching_log = max(training_logs, key=lambda x: x.split('_')[-1].split('.')[0])
    
    return latest_model, step_number, matching_log

def prepare_resume_data():
    """Prepare data needed to resume training."""
    model_path, step_number, log_path = find_latest_checkpoint()
    
    if not model_path:
        return None
    
    print(f"📂 Found checkpoint: {model_path}")
    print(f"📊 Step number: {step_number:,}")
    
    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model checkpoint loaded")
        
        # Extract training state
        resume_data = {
            'model_path': model_path,
            'step_number': step_number,
            'model_state_dict': checkpoint['model_state_dict'],
            'avg_reward': checkpoint.get('avg_reward', 0.0),
            'epsilon': checkpoint.get('epsilon', EPSILON_START),
            'hyperparameters': checkpoint.get('hyperparameters', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        # Load training log if available
        if log_path:
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    resume_data['training_history'] = log_data
                    print(f"📈 Training log loaded: {log_path}")
            except Exception as e:
                print(f"⚠️ Could not load training log: {e}")
                resume_data['training_history'] = None
        
        print(f"📋 Resume data prepared:")
        print(f"   Step: {resume_data['step_number']:,}")
        print(f"   Avg Reward: {resume_data['avg_reward']:.2f}")
        print(f"   Epsilon: {resume_data['epsilon']:.4f}")
        print(f"   Timestamp: {resume_data['timestamp']}")
        
        return resume_data
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def create_resume_training_script():
    """Create a modified training script that resumes from checkpoint."""
    
    # Check if we have resume data
    resume_data = prepare_resume_data()
    if not resume_data:
        print("❌ Cannot resume - no valid checkpoint found!")
        return False
    
    print(f"\\n🔧 Creating resume training script...")
    
    # Create resume-enabled racegame script
    resume_script = f'''#!/usr/bin/env python3
"""
RESUME TRAINING - Modified racegame.py that continues from checkpoint.
Auto-generated script to resume training from step {resume_data['step_number']:,}
""" 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Standard imports from original racegame.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import pyglet as pg
from pyglet import shapes
from pyglet.gl import *
import random

# Project imports
from training_config import *
from game_settings import GameSettings, GameStatus
from gui import GUI
from user_car import UserCar
from ai_car import AICar
from car import Car
from racetrack import Racetrack
from rlenv import RacegameEnv
from Network import Network
from training_monitor import TrainingMonitor, save_model
import time

# Resume configuration
RESUME_FROM_STEP = {resume_data['step_number']}
RESUME_MODEL_PATH = r"{resume_data['model_path']}"
RESUME_EPSILON = {resume_data['epsilon']}
RESUME_AVG_REWARD = {resume_data['avg_reward']}

print("🔄 RESUMING TRAINING FROM CHECKPOINT")
print("=" * 60)
print(f"📂 Model: {{RESUME_MODEL_PATH}}")
print(f"📊 Step: {{RESUME_FROM_STEP:,}}")
print(f"🎯 Epsilon: {{RESUME_EPSILON:.4f}}")
print(f"🏆 Avg Reward: {{RESUME_AVG_REWARD:.2f}}")
print("=" * 60)

# Game setup (same as original)
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
game_window = pg.window.Window(SCREEN_WIDTH, SCREEN_HEIGHT, caption="Resume Training - Q-Learning Car")
batch = pg.graphics.Batch()

settings = GameSettings()
racetrack = Racetrack(settings, game_window, batch)

user_car = UserCar(settings, game_window, batch)
ai_car = AICar(settings, game_window, batch)

gui = GUI(settings)
rl_env = RacegameEnv(ai_car, render_mode="human")

# Neural Networks - RESUME FROM CHECKPOINT
online_net = Network(rl_env)
target_net = Network(rl_env)

# Load checkpoint weights
print("📥 Loading checkpoint weights...")
try:
    checkpoint = torch.load(RESUME_MODEL_PATH, map_location=DEVICE)
    online_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Checkpoint weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading checkpoint: {{e}}")
    exit(1)

# Optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# Training state - RESUME VALUES
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([RESUME_AVG_REWARD], maxlen=100)
step = RESUME_FROM_STEP  # Resume from checkpoint step
episode_reward = 0.0
epsilon = RESUME_EPSILON  # Resume epsilon value

# Initialize environment
obs, _ = rl_env.reset()

# Training monitor with resume capability
monitor = TrainingMonitor()

# Set game to AI training mode immediately
settings.GAME_STATUS = GameStatus.AI_TRAINING

print(f"🚀 Training will resume from step {{step:,}}")
print(f"🤖 Epsilon (exploration rate): {{epsilon:.4f}}")
print(f"📈 Average reward buffer initialized with: {{RESUME_AVG_REWARD:.2f}}")

# Copy all input handlers and training functions from original racegame.py
# (Same as original file - input handlers, update functions, etc.)

# Input-handlers
@game_window.event
def on_mouse_press(x, y, button, modifiers):
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        racetrack.create_boundary(x, y, button)
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        racetrack.create_goal(x, y, button)

@game_window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        racetrack.create_boundary(x, y, buttons)

@game_window.event  
def on_key_press(symbol, modifiers):
    if symbol == pg.window.key.SPACE:
        if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
            settings.GAME_STATUS = GameStatus.DRAW_GOALS
        elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
            settings.GAME_STATUS = GameStatus.USER_CONTROLS
        elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
            settings.GAME_STATUS = GameStatus.AI_TRAINING
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.on_key_press(symbol, modifiers)

@game_window.event
def on_key_release(symbol, modifiers):
    if settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        user_car.on_key_release(symbol, modifiers)

def save_training_results():
    """Save final training results"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/final_model_{{timestamp}}.pth"
    
    save_model(online_net, final_model_path, {{
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        'step': step,
        'avg_reward': float(np.mean(rew_buffer)) if rew_buffer else 0.0,
        'epsilon': epsilon,
        'hyperparameters': {{
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'buffer_size': BUFFER_SIZE,
            'gamma': GAMMA
        }}
    }})
    
    print(f"\\nSaving training results...")
    print(f"Model saved to {{final_model_path}}")
    
    monitor.save_training_log(timestamp, step, rew_buffer, replay_buffer)
    monitor.plot_training_progress()

# Training function - IDENTICAL to original but starts from resumed step
def rl_train():
    global obs, step, episode_reward, epsilon, rew_buffer, replay_buffer
    
    action = online_net.act(obs, epsilon)
    new_obs, reward, terminated, truncated, info = rl_env.step(action)
    done = terminated or truncated
    
    episode_reward += reward
    
    replay_buffer.append((obs, action, reward, new_obs, done))
    
    obs = new_obs
    
    if done:
        rew_buffer.append(episode_reward)
        episode_reward = 0.0
        obs, _ = rl_env.reset()
    
    if len(replay_buffer) > MIN_REPLAY_SIZE:
        
        indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
        batch = [replay_buffer[i] for i in indices]
        
        states = torch.tensor([e[0] for e in batch], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=DEVICE)
        
        current_q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            monitor.log_step(step, loss.item(), np.mean(rew_buffer) if rew_buffer else 0.0)
    
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    step += 1
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    if step % SAVE_INTERVAL == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f"models/model_step_{{step}}.pth"
        
        save_model(online_net, model_path, {{
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            'step': step,
            'avg_reward': float(np.mean(rew_buffer)) if rew_buffer else 0.0,
            'epsilon': epsilon,
            'hyperparameters': {{
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'buffer_size': BUFFER_SIZE,
                'gamma': GAMMA
            }}
        }})
    
    if step % LOG_INTERVAL == 0:
        avg_reward = np.mean(rew_buffer) if rew_buffer else 0.0
        current_loss = monitor.get_recent_loss()
        print(f"Step {{step:>7}} | Avg Reward: {{avg_reward:>7.2f}} | Epsilon: {{epsilon:.4f}} | Loss: {{current_loss:.4f}}")
    
    if step % PLOT_INTERVAL == 0:
        monitor.plot_training_progress()

@game_window.event
def on_draw():
    game_window.clear()
    
    if settings.GAME_STATUS == GameStatus.DRAW_BOUNDARIES:
        racetrack.draw()
        gui.render()
        gui.draw_instructions("Draw boundaries with mouse. Press SPACE when done.")
        
    elif settings.GAME_STATUS == GameStatus.DRAW_GOALS:
        racetrack.draw()
        gui.render()
        gui.draw_instructions("Draw goal points with mouse. Press SPACE when done.")
        
    elif settings.GAME_STATUS == GameStatus.USER_CONTROLS:
        racetrack.draw()
        user_car.update(racetrack)
        user_car.draw()
        gui.render()
        gui.draw_instructions("Use arrow keys to drive. Press SPACE to start AI training.")
        
    elif settings.GAME_STATUS == GameStatus.AI_TRAINING:
        racetrack.draw()
        ai_car.draw()
        gui.render()
        gui.draw_instructions(f"AI Training - Step: {{step:,}} | Epsilon: {{epsilon:.4f}} | Reward: {{np.mean(rew_buffer) if rew_buffer else 0:.2f}}")
        
        rl_train()

def update(dt):
    pass

if __name__ == "__main__":
    print("\\n🎮 STARTING RESUMED TRAINING SESSION")
    print("Press ESC or close window to stop and save results")
    
    try:
        pg.clock.schedule_interval(update, 1/60.0)
        pg.app.run()
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Training interrupted by user.")
        save_training_results()
        print("✅ Results saved!")
    except Exception as e:
        print(f"\\n❌ Error during training: {{e}}")
        save_training_results()
        print("✅ Results saved despite error!")
'''
    
    # Save the resume script
    with open("resume_racegame.py", "w", encoding="utf-8") as f:
        f.write(resume_script)
    
    print(f"✅ Resume script created: resume_racegame.py")
    return True

def main():
    """Main function to set up and start resume training."""
    print("🔄 Q-LEARNING CAR - RESUME TRAINING SETUP")
    print("=" * 60)
    
    if create_resume_training_script():
        print(f"\\n🚀 READY TO RESUME TRAINING!")
        print("=" * 40)
        print(f"\\n📋 TO CONTINUE TRAINING:")
        print(f"   1. Run: python resume_racegame.py")
        print(f"   2. Training will start immediately in AI mode")
        print(f"   3. No need to draw track - will use existing setup")
        print(f"\\n💡 TIPS:")
        print(f"   • Training continues from where you left off")
        print(f"   • Epsilon and rewards are preserved")
        print(f"   • New models will be saved starting from current step")
        print(f"   • Press ESC or close window to stop and save")
        print(f"\\n" + "=" * 40)
        
        # Ask if user wants to start immediately
        choice = input("\\n🎯 Start resumed training now? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            print("\\n🚀 Starting resumed training...")
            import subprocess
            subprocess.run(["python", "resume_racegame.py"])
    else:
        print(f"\\n❌ Could not create resume script. Check if you have valid checkpoints.")
        print(f"\\n💡 To start fresh training, use: python start_training.py")

if __name__ == "__main__":
    main()