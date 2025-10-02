"""
Quick test script to verify the Q-Learning Car setup.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("🔧 TESTING Q-Learning Car Setup...")

# Test 1: Import all modules
try:
    import torch
    import numpy as np
    import pyglet
    import gymnasium
    print("✅ All basic imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test 2: CUDA availability
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name()}")

# Test 3: Try importing project modules
try:
    from training_config import *
    from rlenv import RacegameEnv
    from Network import Network
    print("✅ Project modules imported successfully")
except ImportError as e:
    print(f"❌ Project import error: {e}")

# Test 4: Check if we can create basic objects
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")
    
    # Test tensor operations
    test_tensor = torch.randn(10, 8, device=device)
    print(f"✅ Tensor operations work on {device}")
    
except Exception as e:
    print(f"❌ Device/tensor error: {e}")

print("\n" + "="*50)
print("🚀 READY TO START TRAINING!")
print("Run: python start_training.py")
print("="*50)