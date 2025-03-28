

class SimpleConfig:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory_size = 10000
        self.batch_size = 64
        self.train_start = 1000
