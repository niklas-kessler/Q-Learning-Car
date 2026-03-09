#!/usr/bin/env python3
"""
Quick Q-Values analysis tool for current training state.
"""
import os
import sys
# Ensure project root is on the path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from training.network import Network
from training.training_config import *
import glob

def analyze_q_values():
    """Analyze current Q-values from the latest model."""
    print("Q-VALUES ANALYSIS")
    print("=" * 60)

    # Find latest model
    model_files = glob.glob("models/model_step_*.pth")
    if not model_files:
        print("No model files found. Training has not saved any models yet.")
        return

    # Get latest model by step number
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    step_number = int(latest_model.split('_')[-1].split('.')[0])

    print(f"Loading model: {latest_model}")
    print(f"Training step: {step_number:,}")

    # Create network
    try:
        # Create dummy environment for network initialization
        class MockEnv:
            def __init__(self):
                self.action_space = type('', (), {'n': 8})()

        network = Network(MockEnv())

        # Load trained weights
        checkpoint = torch.load(latest_model, map_location=DEVICE)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()

        print("Model loaded successfully")
        print("Model metadata:")
        for key, value in checkpoint.items():
            if key != 'model_state_dict':
                print(f"   {key}: {value}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test Q-values with different scenarios
    print("\nQ-VALUES ANALYSIS:")
    print("=" * 40)

    # Scenario 1: All sensors at max distance (safe situation)
    safe_obs = [100.0] * 8  # All sensors see max distance
    safe_q = network(torch.tensor(safe_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    print("\nSAFE SITUATION (all sensors = 100):")
    print(f"   Q-values: {safe_q.squeeze().tolist()}")
    print(f"   Best action: {torch.argmax(safe_q).item()} (0=forward, 1=forward-right, etc.)")
    print(f"   Q-value range: [{torch.min(safe_q):.3f}, {torch.max(safe_q):.3f}]")

    # Scenario 2: Wall directly ahead
    wall_ahead_obs = [10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    wall_q = network(torch.tensor(wall_ahead_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    print("\nWALL AHEAD (front sensor = 10):")
    print(f"   Q-values: {wall_q.squeeze().tolist()}")
    print(f"   Best action: {torch.argmax(wall_q).item()}")
    print(f"   Q-value range: [{torch.min(wall_q):.3f}, {torch.max(wall_q):.3f}]")

    # Scenario 3: Surrounded by walls
    trapped_obs = [20.0] * 8  # All sensors see nearby walls
    trapped_q = network(torch.tensor(trapped_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    print("\nSURROUNDED (all sensors = 20):")
    print(f"   Q-values: {trapped_q.squeeze().tolist()}")
    print(f"   Best action: {torch.argmax(trapped_q).item()}")
    print(f"   Q-value range: [{torch.min(trapped_q):.3f}, {torch.max(trapped_q):.3f}]")

    # Scenario 4: Random realistic observations
    print("\nRANDOM SCENARIOS:")
    for i in range(3):
        random_obs = np.random.uniform(10, 100, 8).tolist()
        random_q = network(torch.tensor(random_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        best_action = torch.argmax(random_q).item()
        q_std = torch.std(random_q).item()
        print(f"   Scenario {i+1}: Best action={best_action}, Q-std={q_std:.3f}, Sensors=[{random_obs[0]:.1f}, {random_obs[1]:.1f}, ..., {random_obs[-1]:.1f}]")

    # Q-value statistics
    print("\nQ-VALUE STATISTICS:")
    all_scenarios = [safe_obs, wall_ahead_obs, trapped_obs]
    all_q_values = []

    for obs in all_scenarios:
        q_vals = network(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        all_q_values.extend(q_vals.squeeze().tolist())

    print(f"   Overall Q-range: [{min(all_q_values):.3f}, {max(all_q_values):.3f}]")
    print(f"   Q-value std: {np.std(all_q_values):.3f}")
    print(f"   Mean Q-value: {np.mean(all_q_values):.3f}")

    # Action preferences
    action_names = ["Forward", "Forward-Right", "Right", "Back-Right",
                   "Backward", "Back-Left", "Left", "Forward-Left"]

    action_counts = {}
    for obs in all_scenarios:
        q_vals = network(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        best_action = torch.argmax(q_vals).item()
        action_counts[best_action] = action_counts.get(best_action, 0) + 1

    print("\nACTION PREFERENCES:")
    for action_id, count in action_counts.items():
        print(f"   {action_names[action_id]} ({action_id}): {count} times preferred")

    # Network confidence analysis
    print("\nNETWORK CONFIDENCE:")
    confidences = []
    for obs in all_scenarios:
        q_vals = network(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        max_q = torch.max(q_vals)
        second_max_q = torch.topk(q_vals, 2)[0][0][1]
        confidence = (max_q - second_max_q).item()
        confidences.append(confidence)

    avg_confidence = np.mean(confidences)
    print(f"   Average decision confidence: {avg_confidence:.3f}")
    if avg_confidence > 5.0:
        print("   High confidence - Network has strong preferences")
    elif avg_confidence > 1.0:
        print("   Medium confidence - Network is learning")
    else:
        print("   Low confidence - Network still exploring")

    print("\nQ-Values analysis complete.")

if __name__ == "__main__":
    analyze_q_values()
