"""
Training monitoring and visualization utilities for Q-Learning Car project.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import json
import os
from datetime import datetime
import torch


class TrainingMonitor:
    def __init__(self, save_dir="training_logs"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.episode_rewards = []
        self.losses = []
        self.epsilon_values = []
        self.steps = []
        
        self.avg_rewards_100 = []
        self.max_rewards = []
        
        # Create timestamp for this training session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_step(self, step, episode_reward, loss, epsilon, avg_reward_100=None):
        """Log training metrics for a single step."""
        self.steps.append(step)
        self.episode_rewards.append(episode_reward)
        self.losses.append(loss)
        self.epsilon_values.append(epsilon)
        
        if avg_reward_100 is not None:
            self.avg_rewards_100.append(avg_reward_100)
            
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics = {
            'session_id': self.session_id,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'epsilon_values': self.epsilon_values,
            'avg_rewards_100': self.avg_rewards_100
        }
        
        filename = os.path.join(self.save_dir, f"training_metrics_{self.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filename}")
        
    def plot_training_progress(self, save_plot=True):
        """Create and display training progress plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Session {self.session_id}', fontsize=16)
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.steps, self.episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
        if len(self.avg_rewards_100) > 0:
            axes[0, 0].plot(self.steps[-len(self.avg_rewards_100):], self.avg_rewards_100, 
                           color='red', linewidth=2, label='Avg 100 Episodes')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Loss
        axes[0, 1].plot(self.steps, self.losses, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True)
        
        # Plot 3: Epsilon (Exploration Rate)
        axes[1, 0].plot(self.steps, self.epsilon_values, color='green')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].grid(True)
        
        # Plot 4: Moving Average of Rewards
        if len(self.episode_rewards) > 50:
            window_size = min(50, len(self.episode_rewards))
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(self.steps[window_size-1:], moving_avg, color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Moving Average Reward')
            axes[1, 1].set_title(f'Moving Average Reward (window={window_size})')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = os.path.join(self.save_dir, f"training_progress_{self.session_id}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_filename}")
            
        plt.show()
        
    def print_summary(self):
        """Print a summary of training progress."""
        if not self.episode_rewards:
            print("No training data available yet.")
            return
            
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Session ID: {self.session_id}")
        print(f"Total Steps: {len(self.steps)}")
        print(f"Episodes Completed: {len(self.episode_rewards)}")
        
        if self.episode_rewards:
            print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Best Episode Reward: {max(self.episode_rewards):.2f}")
            print(f"Worst Episode Reward: {min(self.episode_rewards):.2f}")
            
        if self.avg_rewards_100:
            print(f"Current Avg (last 100): {self.avg_rewards_100[-1]:.2f}")
            
        if self.losses:
            print(f"Current Loss: {self.losses[-1]:.4f}")
            print(f"Average Loss: {np.mean(self.losses):.4f}")
            
        if self.epsilon_values:
            print(f"Current Epsilon: {self.epsilon_values[-1]:.3f}")
            
        print("="*50)


def save_model(model, filepath, metadata=None):
    """Save a PyTorch model with optional metadata."""
    save_data = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        save_data.update(metadata)
        
    torch.save(save_data, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Load a PyTorch model from file."""
    import torch
    save_data = torch.load(filepath)
    model.load_state_dict(save_data['model_state_dict'])
    
    metadata = {k: v for k, v in save_data.items() if k != 'model_state_dict'}
    return metadata