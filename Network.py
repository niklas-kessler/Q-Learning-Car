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

        # TODO: dynamic
        in_features = 8  # no. input features

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)  # calculate network output for input? obligatory, not important?

    # choose action for max Q
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))  # calculate network output for input? given s/obs returns q value for each action # unsqueeze(0) to add dimension (for batch) in beginning

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

