"""
Training monitoring and visualization utilities for Q-Learning Car project.
"""
import matplotlib
# WICHTIG: Backend MUSS vor anderen matplotlib imports gesetzt werden
matplotlib.use('Agg')  # Non-blocking backend

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
        
        # Create plots directory
        if not os.path.exists("plots"):
            os.makedirs("plots")
            
        self.episode_rewards = []
        self.losses = []
        self.epsilon_values = []
        self.steps = []
        
        self.avg_rewards_100 = []
        self.max_rewards = []
        
        # Create timestamp for this training session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure matplotlib for non-blocking plots
        plt.ioff()  # Turn off interactive mode
        print(f"📊 TrainingMonitor initialized - Session: {self.session_id}")
        
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
        """Create and SAVE training progress plots without blocking."""
        if not self.losses and not self.episode_rewards:
            print("⚠️  No data to plot yet")
            return
            
        try:
            # Clear any existing plots
            plt.close('all')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Session {self.session_id}', fontsize=16)
            
            # Plot 1: Training Loss (TOP LEFT)
            if self.losses:
                axes[0, 0].plot(self.steps, self.losses, alpha=0.7, color='red', linewidth=1)
                
                # Moving average für Loss mit exakter Längen-Kontrolle
                if len(self.losses) > 50:
                    window = min(100, len(self.losses) // 10)
                    # Use 'same' mode and ensure exact length matching
                    moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='same')
                    # Ensure exact length match (trim if necessary)
                    if len(moving_avg) != len(self.steps):
                        moving_avg = moving_avg[:len(self.steps)]
                    axes[0, 0].plot(self.steps, moving_avg, color='darkred', linewidth=2, label=f'Moving Avg ({window})')
                    axes[0, 0].legend()
                    
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss Over Time')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add current loss value
                if self.losses:
                    current_loss = self.losses[-1]
                    axes[0, 0].text(0.02, 0.98, f'Current Loss: {current_loss:.4f}', 
                                   transform=axes[0, 0].transAxes, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[0, 0].text(0.5, 0.5, 'No Loss Data\n(Training not started yet)', 
                               ha='center', va='center', transform=axes[0, 0].transAxes,
                               fontsize=14, color='red')
                axes[0, 0].set_title('Training Loss - No Data Yet')
            
            # Plot 2: Episode Rewards (TOP RIGHT) 
            if self.episode_rewards:
                # Plot individual episode rewards
                episode_numbers = list(range(1, len(self.episode_rewards) + 1))
                axes[0, 1].plot(episode_numbers, self.episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
                
                # Moving average für Rewards mit exakter Längen-Kontrolle
                if len(self.episode_rewards) > 10:
                    window = min(50, len(self.episode_rewards) // 5)
                    # Use 'same' mode and ensure exact length matching
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='same')
                    # Ensure exact length match (trim if necessary)
                    if len(moving_avg) != len(episode_numbers):
                        moving_avg = moving_avg[:len(episode_numbers)]
                    axes[0, 1].plot(episode_numbers, moving_avg, color='darkblue', linewidth=2, label=f'Moving Avg ({window})')
                    
                axes[0, 1].set_xlabel('Episode Number')
                axes[0, 1].set_ylabel('Total Reward')
                axes[0, 1].set_title('Episode Rewards Over Time')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add statistics
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                axes[0, 1].text(0.02, 0.98, f'Recent Avg (last 10): {recent_avg:.1f}', 
                               transform=axes[0, 1].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[0, 1].text(0.5, 0.5, 'No Episode Data\n(No episodes completed yet)', 
                               ha='center', va='center', transform=axes[0, 1].transAxes,
                               fontsize=14, color='blue')
                axes[0, 1].set_title('Episode Rewards - No Data Yet')
            
            # Plot 3: Epsilon (Exploration Rate) (BOTTOM LEFT)
            if self.epsilon_values:
                axes[1, 0].plot(self.steps, self.epsilon_values, color='green', linewidth=2)
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Epsilon')
                axes[1, 0].set_title('Exploration Rate (Epsilon)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add current epsilon
                current_epsilon = self.epsilon_values[-1]
                axes[1, 0].text(0.02, 0.98, f'Current ε: {current_epsilon:.3f}', 
                               transform=axes[1, 0].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1, 0].text(0.5, 0.5, 'No Epsilon Data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes,
                               fontsize=14, color='green')
                axes[1, 0].set_title('Exploration Rate - No Data Yet')
            
            # Plot 4: Training Statistics (BOTTOM RIGHT)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Training Statistics')
            
            stats_text = []
            stats_text.append(f"Session ID: {self.session_id}")
            
            if self.steps:
                stats_text.append(f"Total Steps: {max(self.steps):,}")
            else:
                stats_text.append("Total Steps: 0")
                
            if self.episode_rewards:
                stats_text.append(f"Episodes Completed: {len(self.episode_rewards):,}")
                stats_text.append(f"Best Episode: {max(self.episode_rewards):.1f}")
                stats_text.append(f"Worst Episode: {min(self.episode_rewards):.1f}")
                stats_text.append(f"Average Reward: {np.mean(self.episode_rewards):.1f}")
            else:
                stats_text.append("Episodes Completed: 0")
                stats_text.append("No episode data yet")
                
            if self.losses:
                stats_text.append(f"Current Loss: {self.losses[-1]:.4f}")
                stats_text.append(f"Average Loss: {np.mean(self.losses):.4f}")
            else:
                stats_text.append("No loss data yet")
                
            # GPU Information
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_name = torch.cuda.get_device_name()
                stats_text.append(f"GPU: {gpu_name}")
                stats_text.append(f"GPU Memory: {gpu_memory:.2f} GB")
            else:
                stats_text.append("Device: CPU")
            
            # Display statistics
            stats_str = "\n".join(stats_text)
            axes[1, 1].text(0.05, 0.95, stats_str, transform=axes[1, 1].transAxes, 
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"plots/training_progress_{timestamp}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"📊 Plot saved to {plot_filename}")
                
            # WICHTIG: Figure schließen um Memory zu sparen
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Error creating plot: {e}")
            import traceback
            traceback.print_exc()
            # Alle Figures schließen bei Fehlern
            plt.close('all')
        
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