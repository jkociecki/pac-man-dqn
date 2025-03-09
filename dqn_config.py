class DQNConfig:
    def __init__(self):
        # Memory and learning parameters
        self.memory_size = 100000
        self.prioritized_replay = True
        self.gamma = 0.99
        self.learning_rate = 0.0003
        self.batch_size = 32
        self.train_start = 1000

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.use_noisy_nets = False

        # Update frequencies
        self.update_target_freq = 1000
        self.model_save_freq = 5000

        # Algorithm enhancements
        self.use_double_dqn = False
        self.use_dueling_dqn = False
        self.n_step_returns = 3