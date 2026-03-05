"""
Entry point for Q-Learning Car training.
Run this script to launch the game and start training.
"""
import os

# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from training.training_config import *


def check_environment():
    """Print environment info and confirm setup before training."""
    print("=" * 50)
    print("Q-LEARNING CAR - ENVIRONMENT CHECK")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    print(f"Selected device: {DEVICE}")

    print("=" * 50)
    print("TRAINING HYPERPARAMETERS")
    print("=" * 50)
    print(f"Learning Rate:      {LEARNING_RATE}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"Buffer Size:        {BUFFER_SIZE}")
    print(f"Min Replay Size:    {MIN_REPLAY_SIZE}")
    print(f"Gamma (Discount):   {GAMMA}")
    print(f"Epsilon Start:      {EPSILON_START}")
    print(f"Epsilon End:        {EPSILON_END}")
    print(f"Epsilon Decay:      {EPSILON_DECAY}")
    print(f"Network Layers:     {NETWORK_HIDDEN_LAYERS}")
    print("=" * 50)

    return DEVICE


def start_training():
    """Launch the game window and start the training loop."""
    check_environment()

    print("\n" + "=" * 60)
    print("Q-LEARNING CAR - SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\n1. DRAW TRACK BOUNDARIES:")
    print("   - The game opens in 'Draw Boundaries' mode")
    print("   - Click to place boundary points with the mouse")
    print("   - Press SPACE to advance to the next mode")
    print("\n2. DRAW GOAL LINES:")
    print("   - In 'Draw Goals' mode, click to place goal lines")
    print("   - The agent will learn to drive through these checkpoints")
    print("   - Press SPACE to advance")
    print("\n3. TEST MANUALLY (optional):")
    print("   - 'User Controls' mode lets you drive with arrow keys")
    print("   - Press SPACE to start AI training")
    print("\n4. AI TRAINING:")
    print("   - The agent trains automatically in real time")
    print(f"   - Console logs every {LOG_FREQ:,} steps")
    print(f"   - Plots saved every {PLOT_FREQ:,} steps")
    print(f"   - Checkpoints saved every {SAVE_FREQ:,} steps")
    print("\nTip: A simple oval track works best for initial training.")
    print("\n" + "=" * 60)
    print("\nLaunching game window...")

    try:
        import racegame
        import pyglet as pg

        print("\nGame loaded. Draw your track, then press SPACE to begin training.")
        print("\nControls:")
        print("  SPACE         - Next mode")
        print("  Mouse         - Draw (in draw modes)")
        print("  Arrow keys    - Drive (in User Controls)")
        print("  Right-click   - Reset environment (during AI training)")
        print("  ESC / close   - Exit")

        pg.app.run()

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
        print("Saving training results...")
        if 'racegame' in locals():
            try:
                racegame.save_training_results()
                print("Results saved.")
            except Exception as e:
                print(f"Could not save results: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    start_training()
