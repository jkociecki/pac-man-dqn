import numpy as np
import torch
import torch.nn as nn


class ConvDQN(nn.Module):
    def __init__(self, input_channels, action_size, dueling=False):
        super(ConvDQN, self).__init__()
        self.dueling = dueling

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_size = 128

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )

    def forward(self, x, state_features=None):
        x = self.conv_layers(x)

        if state_features is not None:
            x = torch.cat([x, state_features], dim=1)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc(x)


class HybridDQN(nn.Module):
    def __init__(self, matrix_channels, feature_size, action_size, dueling=False):
        super(HybridDQN, self).__init__()
        self.dueling = dueling
        self.action_size = action_size  # Dodaj to

        self.matrix_channels = matrix_channels
        self.feature_size = feature_size

        # Convolutional part for matrix representation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(matrix_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU()
        )

        self.conv_out_size = None
        self.combined_size = None

        self.value_stream = None
        self.advantage_stream = None
        self.fc = None

    def initialize_fc_layers(self):
        if self.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(self.combined_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(self.combined_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_size)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.combined_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_size)
            )

    def forward(self, matrix_input, feature_input):
        if self.combined_size is None:
            with torch.no_grad():
                x1 = self.conv_layers(matrix_input)
                x2 = self.feature_fc(feature_input)

                self.conv_out_size = x1.size(1)
                self.combined_size = self.conv_out_size + 64

                self.initialize_fc_layers()

                device = matrix_input.device
                if self.dueling:
                    self.value_stream = self.value_stream.to(device)
                    self.advantage_stream = self.advantage_stream.to(device)
                else:
                    self.fc = self.fc.to(device)

        x1 = self.conv_layers(matrix_input)
        x2 = self.feature_fc(feature_input)

        x = torch.cat([x1, x2], dim=1)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.max_priority if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return [], [], []

        # Increase beta for better importance sampling correction
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calculate probabilities based on priorities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Retrieve experiences
        batch = [self.buffer[idx] for idx in indices]
        states = [s[0] for s in batch]
        actions = np.array([s[1] for s in batch])
        rewards = np.array([s[2] for s in batch])
        next_states = [s[3] for s in batch]
        dones = np.array([s[4] for s in batch])

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)
