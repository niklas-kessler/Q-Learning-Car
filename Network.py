from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt


class Network(nn.Module):

    def __init__(self, env):
        super().__init__()
        
        from training_config import DEVICE, INPUT_SIZE, NETWORK_HIDDEN_LAYERS, DROPOUT_RATE
        
        self.device = DEVICE
        print(f"Network initialized on device: {self.device}")

        # Build network architecture from config
        layers = []
        prev_size = INPUT_SIZE
        
        for hidden_size in NETWORK_HIDDEN_LAYERS:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, env.action_space.n))
        
        self.net = nn.Sequential(*layers)
        
        # Move network to device
        self.to(self.device)
        
        # Initialize weights properly to prevent NaN issues
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights to prevent numerical issues."""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # Ensure input is on the same device as the network
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return self.net(x)

    # choose action for max Q with epsilon-greedy exploration
    def act(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            # Random action for exploration
            return random.randint(0, self.net[-1].out_features - 1)
        
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # Check for invalid observations and return random action as fallback
        if torch.any(torch.isnan(obs_t)) or torch.any(torch.isinf(obs_t)):
            return random.randint(0, self.net[-1].out_features - 1)
        
        q_values = self(obs_t.unsqueeze(0))
        
        # Check Q-values and return random action as fallback
        if torch.any(torch.isnan(q_values)) or torch.any(torch.isinf(q_values)):
            return random.randint(0, self.net[-1].out_features - 1)
        
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

