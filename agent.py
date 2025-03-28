import torch
import torch.nn as nn
from collections import deque
import random


class SimpleDQN(nn.Module):
    def __init__(self, matrix_channels, feature_size, action_size):
        super(SimpleDQN, self).__init__()
        self.action_size = action_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(matrix_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU()
        )

        self.fc = None
        self.conv_out_size = None
        self.combined_size = None

    def initialize_fc_layers(self):
        self.fc = nn.Sequential(
            nn.Linear(self.combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def forward(self, matrix_input, feature_input):
        if self.combined_size is None:
            with torch.no_grad():
                x1 = self.conv_layers(matrix_input)
                x2 = self.feature_fc(feature_input)

                self.conv_out_size = x1.size(1)
                self.combined_size = self.conv_out_size + 32

                self.initialize_fc_layers()

                device = matrix_input.device
                self.fc = self.fc.to(device)

        x1 = self.conv_layers(matrix_input)
        x2 = self.feature_fc(feature_input)

        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)