# Q-Learning-Car
 
This project aims to implement a simple 2d car-racegame and then use the Deep-Q-Learning Algorithm to train an AI-Agent on it.

Starting point: start_training.py

### Pre Training:

For checking and setting hyperparameters see training_config.py. 

### During Training:

Updates via:
- **Logs**: Every 1000 Steps Progress-Updates
- **Plots**: Every 5000 Steps Visualisations
- **Models**: Every 10000 Steps saved Checkpoints

### After Training:

Repository structure:
```
├── training_logs/          # All Trainings-Metrics
├── models/                 # Saved Models
├── training_progress_*.png # Progress-Plots
└── training_metrics_*.json # Raw data
```