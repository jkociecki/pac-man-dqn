import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pygame
from dqn_config import SimpleConfig
from game.enviroment import SimplePacmanEnv
from game.game_core import Direction
from agent import SimpleDQN, SimpleReplayBuffer


class SimpleDQNAgent:
    def __init__(self, state_matrix_shape, state_feature_size, action_size, config=None):
        self.state_matrix_shape = state_matrix_shape
        self.state_feature_size = state_feature_size
        self.action_size = action_size
        self.action_space = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        self.config = config if config else SimpleConfig()

        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = SimpleReplayBuffer(self.config.memory_size)

        self.model = SimpleDQN(state_matrix_shape[2], state_feature_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_step = 0

    def _preprocess_state(self, state_dict):
        matrix = torch.FloatTensor(np.transpose(state_dict["matrix"], (2, 0, 1))).to(self.device)
        features = torch.FloatTensor(state_dict["features"]).to(self.device)
        return matrix, features

    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.index(action)
        self.memory.push(state, action_index, reward, next_state, done)

    def get_action(self, state, eval_mode=False):
        eps = 0.05 if eval_mode else self.epsilon

        if np.random.random() < eps:
            return random.choice(self.action_space)

        matrix, features = self._preprocess_state(state)
        matrix_tensor = torch.FloatTensor(matrix).unsqueeze(0).to(self.device)
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(matrix_tensor, feature_tensor)
        self.model.train()

        action_index = torch.argmax(q_values).item()
        return self.action_space[action_index]

    def _prepare_batch(self, states, next_states):
        matrix_batch = []
        feature_batch = []
        next_matrix_batch = []
        next_feature_batch = []

        for state in states:
            matrix, features = self._preprocess_state(state)
            matrix_batch.append(matrix)
            feature_batch.append(features)

        for next_state in next_states:
            next_matrix, next_features = self._preprocess_state(next_state)
            next_matrix_batch.append(next_matrix)
            next_feature_batch.append(next_features)

        matrix_tensor = torch.FloatTensor(np.array(matrix_batch)).to(self.device)
        feature_tensor = torch.FloatTensor(np.array(feature_batch)).to(self.device)
        next_matrix_tensor = torch.FloatTensor(np.array(next_matrix_batch)).to(self.device)
        next_feature_tensor = torch.FloatTensor(np.array(next_feature_batch)).to(self.device)

        return matrix_tensor, feature_tensor, next_matrix_tensor, next_feature_tensor

    def _preprocess_state(self, state_dict):
        matrix = torch.FloatTensor(np.transpose(state_dict["matrix"], (2, 0, 1))).to(self.device)
        features = torch.FloatTensor(state_dict["features"]).to(self.device)
        return matrix, features

    def get_action(self, state, eval_mode=False):
        eps = 0.05 if eval_mode else self.epsilon

        if np.random.random() < eps:
            return random.choice(self.action_space)

        matrix, features = self._preprocess_state(state)
        matrix_tensor = matrix.unsqueeze(0)
        feature_tensor = features.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(matrix_tensor, feature_tensor)

        action_index = torch.argmax(q_values).item()
        return self.action_space[action_index]

    def train(self):
        if len(self.memory) < self.config.train_start:
            return 0

        minibatch = self.memory.sample(self.batch_size)

        states, action_indices, rewards, next_states, dones = zip(*minibatch)

        matrices = []
        features = []
        next_matrices = []
        next_features = []

        for state in states:
            matrix, feature = self._preprocess_state(state)
            matrices.append(matrix)
            features.append(feature)

        for next_state in next_states:
            next_matrix, next_feature = self._preprocess_state(next_state)
            next_matrices.append(next_matrix)
            next_features.append(next_feature)

        matrix_tensor = torch.stack(matrices)
        feature_tensor = torch.stack(features)
        next_matrix_tensor = torch.stack(next_matrices)
        next_feature_tensor = torch.stack(next_features)
        action_tensor = torch.LongTensor(action_indices).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(matrix_tensor, feature_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.model(next_matrix_tensor, next_feature_tensor).max(1)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1

        return loss.item()

    def save(self, name="pacman_simple_dqn_model.pth"):
        self.model.to("cpu")
        try:
            directory = os.path.dirname(name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_step': self.train_step,
                'epsilon': self.epsilon
            }

            torch.save(save_dict, name)
            print(f"Model saved successfully to: {name}")

        except Exception as e:
            print(f"Error saving model: {str(e)}")

        self.model.to(self.device)

    def load(self, name="pacman_simple_dqn_model.pth"):
        if not os.path.exists(name):
            print(f"Model file does not exist: {name}")
            return False

        try:
            checkpoint = torch.load(name, map_location=self.device)

            state_dict = checkpoint['model_state_dict']
            remove_prefix = 'module.'
            state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                          state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)

            if hasattr(self.model, 'combined_size') and self.model.combined_size is None:
                dummy_matrix = torch.zeros((1, self.state_matrix_shape[2],
                                            self.state_matrix_shape[0],
                                            self.state_matrix_shape[1])).to(self.device)
                dummy_features = torch.zeros((1, self.state_feature_size)).to(self.device)
                _ = self.model(dummy_matrix, dummy_features)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Could not load optimizer state - initializing new optimizer")

            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']

            print(f"Model loaded successfully from: {name}")
            print(f"Current epsilon: {self.epsilon}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def train_pacman_dqn(episodes=5000, render=True, load_model=False, model_path="models/pacman_simple_dqn_model.pth",
                     save_interval=20, eval_interval=500, fps=30):
    env = SimplePacmanEnv(render=render, fps=fps)

    initial_state = env.reset()

    state_matrix_shape = initial_state["matrix"].shape
    state_feature_size = len(initial_state["features"])

    agent = SimpleDQNAgent(
        state_matrix_shape=state_matrix_shape,
        state_feature_size=state_feature_size,
        action_size=4
    )

    if load_model:
        success = agent.load(model_path)
        if not success:
            print("Failed to load model, starting with a new one.")

    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    avg_losses = []
    best_score = 0

    progress_bar = tqdm(range(1, episodes + 1), desc="Training Progress")

    for episode in progress_bar:
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = []
        episode_length = 0

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        while not done:
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            loss = agent.train()
            if loss > 0:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

        episode_rewards.append(episode_reward)
        episode_scores.append(info["score"])
        episode_lengths.append(episode_length)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        avg_losses.append(avg_loss)

        progress_bar.set_postfix({
            "Reward": f"{episode_reward:.1f}",
            "Score": info["score"],
            "Steps": episode_length,
            "Loss": f"{avg_loss:.4f}",
            "Epsilon": f"{agent.epsilon:.4f}"
        })

        if episode % save_interval == 0:
            agent.save(model_path)
            print(f"\nModel saved after episode {episode}")

            if info["score"] > best_score:
                best_score = info["score"]
                agent.save("models/pacman_dqn_best.pth")
                print(f"New best score: {best_score}, saved as best model")

        if episode % eval_interval == 0:
            eval_score = evaluate_agent(agent, env, num_episodes=3, render=render)
            print(f"\nEvaluation after {episode} episodes: Average Score = {eval_score:.1f}")

    agent.save(model_path)
    print("Final model saved")
    env.close()

    return {
        "episode_rewards": episode_rewards,
        "episode_scores": episode_scores,
        "episode_lengths": episode_lengths,
        "avg_losses": avg_losses
    }


def evaluate_agent(agent, env, num_episodes=5, render=True):
    total_scores = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            action = agent.get_action(state, eval_mode=True)
            next_state, _, done, info = env.step(action)
            state = next_state
            episode_score = info["score"]
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return 0
        total_scores.append(episode_score)
    return np.mean(total_scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or play Pac-Man with DQN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes for training")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--load_model", action="store_true", help="Load existing model")
    parser.add_argument("--model_path", type=str, default="models/pacman_dqn_best.pth", help="Path to model file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")
    args = parser.parse_args()

    if not os.path.exists("models"):
        os.makedirs("models")

    if args.mode == "train":
        history = train_pacman_dqn(
            episodes=args.episodes,
            render=args.render,
            load_model=args.load_model,
            model_path=args.model_path,
            fps=args.fps
        )