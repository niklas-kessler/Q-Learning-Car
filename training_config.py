"""
Configuration file for Q-Learning Car training hyperparameters.
Modify these values to tune the training performance.
"""

# Neural Network Architecture
NETWORK_HIDDEN_LAYERS = [128, 128, 64]  # Hidden layer sizes
DROPOUT_RATE = 0.1
ACTIVATION_FUNCTION = "relu"  # "relu", "tanh", "sigmoid"

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
GAMMA = 0.99  # Discount factor

# Exploration Parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 50000  # Steps over which epsilon decays

# Training Schedule
TARGET_UPDATE_FREQ = 1000  # Update target network every N steps
LOG_FREQ = 1000           # Print logs every N steps
SAVE_FREQ = 10000         # Save model every N steps
PLOT_FREQ = 5000          # Generate plots every N steps

# Reward System Configuration
CRASH_PENALTY = -100      # Penalty for collision
GOAL_REWARD = 50          # Reward for reaching goal
DISTANCE_REWARD_SCALE = 0.5   # Scale for distance-based rewards
VELOCITY_REWARD_SCALE = 0.1   # Scale for velocity-based rewards
SENSOR_PENALTY_SCALE = 0.3    # Scale for sensor-based penalties
SURVIVAL_REWARD = 0.01        # Small reward for each step survived

# Performance Thresholds
SUCCESS_THRESHOLD = 200   # Average reward threshold for "success"
EARLY_STOPPING_PATIENCE = 20000  # Steps without improvement before stopping

# Hardware Configuration
USE_CUDA = True          # Use GPU if available
DEVICE_ID = 0            # GPU device ID (if multiple GPUs)

# Logging and Visualization
SAVE_TRAINING_LOGS = True
SAVE_MODEL_CHECKPOINTS = True
GENERATE_PLOTS = True
VERBOSE_LOGGING = True