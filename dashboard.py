#!/usr/bin/env python3
"""
Dashboard to monitor training progress and verify continuous loss logging.
"""
import os
# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from datetime import datetime
from training_config import *

def show_training_dashboard():
    """Show a dashboard of recent training sessions."""
    print("📊 TRAINING DASHBOARD")
    print("=" * 60)
    
    # List all training logs
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        print("❌ No training logs found!")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith("training_metrics_") and f.endswith(".json")]
    log_files.sort(reverse=True)  # Most recent first
    
    if not log_files:
        print("❌ No training metrics found!")
        return
    
    print(f"🔍 Found {len(log_files)} training sessions:")
    print()
    
    for i, log_file in enumerate(log_files[:5]):  # Show last 5 sessions
        log_path = os.path.join(log_dir, log_file)
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            session_id = data.get('session_id', 'Unknown')
            steps = data.get('steps', [])
            losses = data.get('losses', [])
            episode_rewards = data.get('episode_rewards', [])
            
            print(f"📈 Session {i+1}: {session_id}")
            print(f"   📊 Total Steps: {len(steps):,}")
            print(f"   📉 Loss Data Points: {len(losses):,}")
            print(f"   🏆 Episode Data Points: {len(episode_rewards):,}")
            
            if losses:
                print(f"   🔍 Loss Range: {min(losses):.4f} - {max(losses):.4f}")
                print(f"   📈 Latest Loss: {losses[-1]:.4f}")
                
                # Check for loss curve continuity
                if len(losses) > 100:
                    recent_trend = np.mean(losses[-100:]) - np.mean(losses[-200:-100]) if len(losses) > 200 else 0
                    trend_emoji = "📉" if recent_trend < 0 else "📈" if recent_trend > 0 else "➡️"
                    print(f"   {trend_emoji} Recent Trend: {recent_trend:+.4f}")
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                print(f"   🎯 Average Reward: {avg_reward:.2f}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            print()
    
    # Show current configuration
    print("⚙️  CURRENT CONFIGURATION:")
    print(f"   📊 Plot Frequency: Every {PLOT_FREQ:,} steps")
    print(f"   📝 Log Frequency: Every {LOG_FREQ:,} steps")
    print(f"   💾 Save Frequency: Every {SAVE_FREQ:,} steps")
    print(f"   🧠 Batch Size: {BATCH_SIZE}")
    print(f"   🎯 Network Architecture: {NETWORK_HIDDEN_LAYERS}")
    print(f"   🖥️  Device: {DEVICE}")
    print()
    
    # Show plots directory
    plots_dir = "plots"
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        plot_files.sort(reverse=True)
        print(f"🎨 RECENT PLOTS ({len(plot_files)} total):")
        for plot_file in plot_files[:3]:  # Show last 3 plots
            print(f"   📊 {plot_file}")
        print()
    
    print("✅ Dashboard complete! Ready for training.")
    print()
    print("🚀 TO START TRAINING:")
    print("   1. Run: python start_training.py")
    print("   2. Draw track boundaries in GUI")
    print("   3. Press SPACE to switch to AI Training")
    print("   4. Watch the loss curves get generated!")

if __name__ == "__main__":
    show_training_dashboard()