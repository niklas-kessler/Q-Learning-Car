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
    print(f"🚀 GPU Training enabled: {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("🖥️  CPU Training (no GPU available)")

# ========================
# NEURAL NETWORK (GPU OPTIMIZED)
# ========================
INPUT_SIZE = 8                    # Sensor inputs
OUTPUT_SIZE = 8                   # Action space size
NETWORK_HIDDEN_LAYERS = [256, 256, 128]  # Increased layer sizes for better GPU utilization
DROPOUT_RATE = 0.1
ACTIVATION_FUNCTION = "relu"      # "relu", "tanh", "sigmoid"

# ========================
# Q-LEARNING PARAMETERS (GPU OPTIMIZED)
# ========================
LEARNING_RATE = 1e-4
BATCH_SIZE = 2048                  # Increased for better GPU utilization (was 64)
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
GAMMA = 0.99                      # Discount factor
TARGET_UPDATE_FREQ = 1000         # Update target network every N steps

# ========================
# EXPLORATION (EPSILON)
# ========================
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 50000             # Steps over which epsilon decays

# ========================
# REWARD SYSTEM
# ========================
CRASH_PENALTY = -100              # Penalty for collision
GOAL_REWARD = 250                  # Reward for reaching goal
DISTANCE_REWARD_SCALE = 0.5       # Scale for distance-based rewards
VELOCITY_REWARD_SCALE = 0.1       # Scale for velocity-based rewards
SENSOR_PENALTY_SCALE = 0.3        # Scale for sensor-based penalties
SURVIVAL_REWARD = 0.01            # Small reward for each step survived

# ========================
# TRAINING SCHEDULE
# ========================
LOG_FREQ = 1000                   # Print logs every N steps
SAVE_FREQ = 10000                 # Save model every N steps
PLOT_FREQ = 2000                  # Generate plots every N steps (reduced for better monitoring)
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
SAVE_PLOTS_ONLY = True            # Speichert Plots, zeigt sie aber nicht an
PLOT_DPI = 150                    # Plot Qualität
PLOT_FIGSIZE = (15, 10)           # Plot Größe

print(f"⚙️  Training config loaded - Device: {DEVICE}")