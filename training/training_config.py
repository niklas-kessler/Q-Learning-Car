"""
CENTRAL Configuration file for Q-Learning Car training.
ALL hyperparameters are defined here - NO duplicates elsewhere!
"""
import torch

# ========================
# HARDWARE CONFIGURATION
# ========================
USE_CUDA = True
DEVICE_ID = 0

# Auto-detect best device
if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{DEVICE_ID}")
    print(f"GPU training enabled: {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("CPU training (no GPU available)")

# ========================
# NEURAL NETWORK (GPU OPTIMIZED)
# ========================
INPUT_SIZE = 11                   # 8 sensor inputs + velocity + angle to goal + distance to goal
OUTPUT_SIZE = 8                   # Action space size
NETWORK_HIDDEN_LAYERS = [128, 64] # Smaller net converges faster for an 8-input problem
DROPOUT_RATE = 0.0                # No dropout - stochastic Q-values break DQN stability
ACTIVATION_FUNCTION = "relu"      # "relu", "tanh", "sigmoid"

# ========================
# Q-LEARNING PARAMETERS (GPU OPTIMIZED)
# ========================
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
GRADIENT_STEPS_PER_FRAME = 4     # Gradient updates per game frame - better GPU use without changing game speed
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
GAMMA = 0.99                      # Discount factor
TARGET_UPDATE_FREQ = 5000         # Update target network every N steps (increased for stability)

# ========================
# EXPLORATION (EPSILON)
# ========================
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 300000            # Steps over which epsilon decays

# ========================
# REWARD SYSTEM
# ========================
CRASH_PENALTY = -100             # Penalty for collision
GOAL_REWARD = 250                  # Reward for reaching goal
DISTANCE_REWARD_SCALE = 0.05  # Scale for distance-based rewards
VELOCITY_REWARD_SCALE = 0.01   # Scale for velocity-based rewards
SENSOR_PENALTY_SCALE = 0.03       # Scale for sensor-based penalties
SURVIVAL_REWARD = 0.01            # Small reward for each step survived

# ========================
# TRAINING SCHEDULE
# ========================
LOG_FREQ = 4000                   # Print logs every N gradient steps (~1000 game frames)
SAVE_FREQ = 40000                 # Save model every N gradient steps (~10000 game frames)
PLOT_FREQ = 8000                  # Generate plots every N gradient steps (~2000 game frames)
MAX_STEPS = 1000000               # Maximum training steps

# ========================
# PERFORMANCE THRESHOLDS
# ========================
SUCCESS_THRESHOLD = 200           # Average reward threshold for "success"
EARLY_STOPPING_PATIENCE = 20000   # Steps without improvement before stopping

# ========================
# LOGGING & VISUALIZATION
# ========================
SAVE_TRAINING_LOGS = True
SAVE_MODEL_CHECKPOINTS = True
GENERATE_PLOTS = True
VERBOSE_LOGGING = True

# ========================
# PLOT CONFIGURATION (NON-BLOCKING!)
# ========================
PLOT_BACKEND = 'Agg'              # Non-blocking plot backend
SAVE_PLOTS_ONLY = True            # Save plots without displaying them
PLOT_DPI = 150                    # Plot quality (DPI)
PLOT_FIGSIZE = (15, 10)           # Plot size (width, height)

# ========================
# RESUME TRAINING SETTINGS
# ========================
AUTO_RESUME = False                # Automatically resume from latest checkpoint
RESUME_FROM_STEP = None          # None = auto-find, or specific step number
RESUME_MODEL_PATH = None         # None = auto-find, or specific model path

print(f"Training config loaded - Device: {DEVICE}")
if AUTO_RESUME:
    print("Auto-resume enabled - will continue from latest checkpoint if available")