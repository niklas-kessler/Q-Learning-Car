from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt


class Network(nn.Module):

    def __init__(self, env, device=None):
        super().__init__()
        
        # Automatic device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Network initialized on device: {self.device}")

        # TODO: dynamic
        in_features = 8  # no. input features

        # Improved network architecture with better capacity and regularization
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )
        
        # Move network to device
        self.to(self.device)

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
        q_values = self(obs_t.unsqueeze(0))
        
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

