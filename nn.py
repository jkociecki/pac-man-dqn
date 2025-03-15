import gc
import time
from tqdm import tqdm
from main import Pacman, Ghost, Maze, GameState, Bonus, MAZE_LAYOUT
import os
from collections import deque
import torch
from torch import optim
from dqn_config import DQNConfig
from agent import PrioritizedReplayBuffer, HybridDQN
import torch.nn.functional as F
from game_core import *


class AdvancedPacmanEnv:
    def __init__(self, render=True, fps=FPS):
        self.render_enabled = render
        self.fps = fps
        self.screen_width = len(MAZE_LAYOUT[0]) * CELL_SIZE
        self.screen_height = len(MAZE_LAYOUT) * CELL_SIZE

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Advanced Pac-Man DQN Training")
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lengths = []
        self.avg_q_values = []
        self.danger_map = None
        self.reward_map = None

        self.reset()

    def reset(self):
        self.maze = Maze(MAZE_LAYOUT)
        pacman_start_x = self.screen_width // 2
        pacman_start_y = self.screen_height // 4 - CELL_SIZE
        self.pacman = Pacman(pacman_start_x, pacman_start_y)

        self.ghosts = [
            Ghost(self.screen_width // 4 + CELL_SIZE, self.screen_height * 3 // 4, (255, 0, 0)),  # RED
            Ghost(self.screen_width * 3 // 4, self.screen_height * 3 // 4, (255, 105, 180)),  # PINK
            Ghost(self.screen_width // 4, self.screen_height * 3 // 4, (255, 165, 0)),  # ORANGE
            Ghost(self.screen_width * 3 // 4, self.screen_height * 3 // 4, (0, 255, 0))  # GREEN
        ]

        self.bonuses = [
            Bonus(self.screen_width // 2, self.screen_height // 2)
        ]

        self.game_over = False
        self.level_completed = False
        self.previous_score = 0
        self.steps_without_reward = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.previous_dot_count = sum(1 for _, _, active in self.maze.dots if active)
        self.previous_pacman_position = (self.pacman.x, self.pacman.y)
        self.previous_legal_actions = []

        game_state = GameState(self.pacman, self.ghosts, self.maze, self.bonuses)

        self.danger_map = game_state.calculate_danger_map()
        self.reward_map = game_state.calculate_reward_map()

        return game_state.get_observation()

    import gc
    def step(self, action):
        previous_position = (self.pacman.x, self.pacman.y)
        previous_lives = self.pacman.lives
        previous_score = self.pacman.score
        dots_before = sum(1 for _, _, active in self.maze.dots if active)
        power_dots_before = sum(1 for _, _, active in self.maze.power_dots if active)

        game_state_before = GameState(self.pacman, self.ghosts, self.maze, self.bonuses)
        self.previous_legal_actions = game_state_before.get_legal_actions()

        self.pacman.next_direction = action
        self.pacman.update(self.maze)
        self.maze.check_dot_collision(self.pacman)

        if self.total_steps % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dots_after = sum(1 for _, _, active in self.maze.dots if active)
        power_dots_after = sum(1 for _, _, active in self.maze.power_dots if active)
        dots_eaten = dots_before - dots_after
        power_dots_eaten = power_dots_before - power_dots_after

        for ghost in self.ghosts:
            ghost.is_scared = self.pacman.is_powered_up
            ghost.update(self.maze, self.pacman)

        ghost_collision = False
        ghost_eaten = False

        for ghost in self.ghosts:
            distance = np.sqrt((ghost.x - self.pacman.x) ** 2 + (ghost.y - self.pacman.y) ** 2)
            if distance < (ghost.radius + self.pacman.radius) * 0.8:
                if ghost.is_scared:
                    ghost.x = self.screen_width // 2
                    ghost.y = self.screen_height // 2
                    self.pacman.score += 200
                    ghost_eaten = True
                else:
                    self.pacman.lives -= 1
                    ghost_collision = True
                    if self.pacman.lives <= 0:
                        self.game_over = True
                    else:
                        pacman_start_x = self.screen_width // 2
                        pacman_start_y = self.screen_height // 4 - CELL_SIZE
                        self.pacman.x = pacman_start_x
                        self.pacman.y = pacman_start_y
                        self.pacman.direction = Direction.NONE
                        self.pacman.next_direction = Direction.NONE

        bonus_eaten = False
        for bonus in self.bonuses:
            if bonus.active:
                distance = np.sqrt((bonus.x - self.pacman.x) ** 2 + (bonus.y - self.pacman.y) ** 2)
                if distance < (bonus.radius + self.pacman.radius):
                    bonus.active = False
                    self.pacman.score += 100
                    bonus_eaten = True
                    if bonus.type == "cherry":
                        self.pacman.score += 100
                    elif bonus.type == "pill":
                        self.pacman.lives += 1

        if self.maze.all_dots_eaten():
            self.level_completed = True

        reward = self._calculate_reward(
            ghost_collision=ghost_collision,
            ghost_eaten=ghost_eaten,
            dots_eaten=dots_eaten,
            power_dots_eaten=power_dots_eaten,
            bonus_eaten=bonus_eaten,
            previous_position=previous_position,
            previous_lives=previous_lives,
            previous_legal_actions=self.previous_legal_actions,
            action=action
        )

        self.episode_reward += reward

        if reward > 0:
            self.steps_without_reward = 0
        else:
            self.steps_without_reward += 1

        self.total_steps += 1

        game_state = GameState(self.pacman, self.ghosts, self.maze, self.bonuses)

        self.danger_map = game_state.calculate_danger_map()
        self.reward_map = game_state.calculate_reward_map()

        done = (
                self.game_over or
                self.level_completed or
                self.steps_without_reward > 300
        )

        if self.render_enabled:
            self._render()

        new_observation = game_state.get_observation()
        info = {
            "score": self.pacman.score,
            "lives": self.pacman.lives,
            "steps": self.total_steps,
            "dots_eaten": dots_eaten,
            "power_dots_eaten": power_dots_eaten,
        }

        return new_observation, reward, done, info

    def _calculate_reward(self, ghost_collision, ghost_eaten, dots_eaten, power_dots_eaten,
                          bonus_eaten, previous_position, previous_lives, previous_legal_actions, action):
        reward = 0

        reward += dots_eaten * 10
        reward += power_dots_eaten * 50
        reward += bonus_eaten * 25

        if ghost_eaten:
            reward += 100

        if ghost_collision:
            reward -= 200

        if self.level_completed:
            reward += 500

        if self.game_over:
            reward -= 300

        current_position = (self.pacman.x, self.pacman.y)
        if current_position == previous_position:
            reward -= 1

        if self.steps_without_reward > 100:
            reward -= 0.5

        if self.danger_map is not None:
            pacman_cell_x = int(self.pacman.x // CELL_SIZE)
            pacman_cell_y = int(self.pacman.y // CELL_SIZE)
            if 0 <= pacman_cell_x < len(self.danger_map[0]) and 0 <= pacman_cell_y < len(self.danger_map):
                danger_level = self.danger_map[pacman_cell_y][pacman_cell_x]
                if danger_level > 0.7:
                    reward -= 5
                elif danger_level > 0.4:
                    reward -= 2

        if self.reward_map is not None:
            pacman_cell_x = int(self.pacman.x // CELL_SIZE)
            pacman_cell_y = int(self.pacman.y // CELL_SIZE)
            if 0 <= pacman_cell_x < len(self.reward_map[0]) and 0 <= pacman_cell_y < len(self.reward_map):
                reward_potential = self.reward_map[pacman_cell_y][pacman_cell_x]
                # Bonus for moving toward high-value areas
                reward += reward_potential * 2

        if action in previous_legal_actions and len(previous_legal_actions) > 1:
            if action == Direction.UP and Direction.DOWN in previous_legal_actions:
                reward -= 0.5
            elif action == Direction.DOWN and Direction.UP in previous_legal_actions:
                reward -= 0.5
            elif action == Direction.LEFT and Direction.RIGHT in previous_legal_actions:
                reward -= 0.5
            elif action == Direction.RIGHT and Direction.LEFT in previous_legal_actions:
                reward -= 0.5

        return reward

    def _render(self):
        if not self.render_enabled or self.screen is None:
            return

        self.screen.fill((0, 0, 0))

        self.maze.draw(self.screen)

        self.pacman.draw(self.screen)

        for ghost in self.ghosts:
            ghost.draw(self.screen)

        for bonus in self.bonuses:
            if bonus.active:
                bonus.draw(self.screen)

        score_text = self.font.render(f"Score: {self.pacman.score}", True, (255, 255, 255))
        lives_text = self.font.render(f"Lives: {self.pacman.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (self.screen_width - 150, 10))

        if hasattr(self, 'episode_reward'):
            reward_text = self.small_font.render(f"Episode Reward: {self.episode_reward:.1f}", True, (255, 255, 0))
            steps_text = self.small_font.render(f"Steps: {self.total_steps}", True, (255, 255, 0))
            self.screen.blit(reward_text, (10, self.screen_height - 60))
            self.screen.blit(steps_text, (10, self.screen_height - 30))

        pygame.display.flip()

        if self.clock:
            self.clock.tick(self.fps)

    def close(self):
        if self.render_enabled:
            pygame.quit()


def train_pacman_dqn(episodes=10000, render=True, load_model=False, model_path="models/pacman_advanced_dqn_model.pth",
                     save_interval=20, eval_interval=500, fps=30):
    env = AdvancedPacmanEnv(render=render, fps=fps)

    initial_state = env.reset()

    state_matrix_shape = initial_state["matrix"].shape
    state_feature_size = len(initial_state["features"])

    config = DQNConfig()
    agent = AdvancedDQNAgent(
        state_matrix_shape=state_matrix_shape,
        state_feature_size=state_feature_size,
        action_size=4,
        config=config
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

        if render and episode_length % 4 == 0:
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
        if episode % 2 == 0:
            agent.save(model_path)
            print(f"\nModel saved after episode {episode}")
            if info["score"] > best_score:
                best_score = info["score"]
                agent.save("models/pacman_dqn_best.pth")
                print(f"New best score: {best_score}, saved as best model")
        if episode % eval_interval == 0:
            eval_score = evaluate_agent(agent, env, num_episodes=5, render=render)
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


def evaluate_agent(agent, env, num_episodes=10, render=True):
    """Evaluate agent performance over multiple episodes"""
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



# Optimized DQN agent
class AdvancedDQNAgent:
    def __init__(self, state_matrix_shape, state_feature_size, action_size, config=None):
        self.state_matrix_shape = state_matrix_shape
        self.state_feature_size = state_feature_size
        self.action_size = action_size
        self.action_space = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        self.config = config if config else DQNConfig()

        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.use_double_dqn = self.config.use_double_dqn
        self.n_step_returns = self.config.n_step_returns

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.config.memory_size)
        else:
            self.memory = deque(maxlen=self.config.memory_size)

        self.model = HybridDQN(
            state_matrix_shape[2],
            state_feature_size,
            action_size,
            dueling=self.config.use_dueling_dqn
        ).to(self.device)

        self.target_model = HybridDQN(
            state_matrix_shape[2],
            state_feature_size,
            action_size,
            dueling=self.config.use_dueling_dqn
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_step = 0
        self.update_target_counter = 0

        self.update_target_model()

        self.n_step_buffer = deque(maxlen=self.n_step_returns)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _preprocess_state(self, state_dict):
        matrix = np.transpose(state_dict["matrix"], (2, 0, 1))
        features = state_dict["features"]
        return matrix, features

    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.index(action)

        self.n_step_buffer.append((state, action_index, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step_returns:
            return

        R = 0
        n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]

        for i in range(self.n_step_returns):
            R += self.gamma ** i * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                break

        n_step_next_state = next_state
        n_step_done = done

        if self.config.prioritized_replay:
            self.memory.push(n_step_state, n_step_action, R, n_step_next_state, n_step_done)
        else:
            self.memory.append((n_step_state, n_step_action, R, n_step_next_state, n_step_done))

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

    def train(self):

        torch.cuda.empty_cache()

        """Train model on a batch from memory"""
        if self.config.prioritized_replay:
            if len(self.memory) < self.config.train_start:
                return 0

            indices, weights, batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch

            matrix_tensor, feature_tensor, next_matrix_tensor, next_feature_tensor = self._prepare_batch(states,
                                                                                                         next_states)

            action_tensor = torch.LongTensor(actions).to(self.device)
            reward_tensor = torch.FloatTensor(rewards).to(self.device)
            done_tensor = torch.FloatTensor(dones).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            current_q = self.model(matrix_tensor, feature_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

            if self.use_double_dqn:
                next_actions = self.model(next_matrix_tensor, next_feature_tensor).max(1)[1]
                next_q = self.target_model(next_matrix_tensor, next_feature_tensor).gather(1, next_actions.unsqueeze(
                    1)).squeeze(1)
            else:
                next_q = self.target_model(next_matrix_tensor, next_feature_tensor).max(1)[0]

            target_q = reward_tensor + (1 - done_tensor) * self.gamma ** self.n_step_returns * next_q

            td_error = torch.abs(current_q - target_q).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_error)

            loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        else:
            if len(self.memory) < self.config.train_start:
                return 0

            minibatch = random.sample(self.memory, self.batch_size)
            states = [s[0] for s in minibatch]
            actions = np.array([s[1] for s in minibatch])
            rewards = np.array([s[2] for s in minibatch])
            next_states = [s[3] for s in minibatch]
            dones = np.array([s[4] for s in minibatch])

            matrix_tensor, feature_tensor, next_matrix_tensor, next_feature_tensor = self._prepare_batch(states,
                                                                                                         next_states)

            action_tensor = torch.LongTensor(actions).to(self.device)
            reward_tensor = torch.FloatTensor(rewards).to(self.device)
            done_tensor = torch.FloatTensor(dones).to(self.device)

            current_q = self.model(matrix_tensor, feature_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

            if self.use_double_dqn:
                next_actions = self.model(next_matrix_tensor, next_feature_tensor).max(1)[1]
                next_q = self.target_model(next_matrix_tensor, next_feature_tensor).gather(1, next_actions.unsqueeze(
                    1)).squeeze(1)
            else:
                next_q = self.target_model(next_matrix_tensor, next_feature_tensor).max(1)[0]

            target_q = reward_tensor + (1 - done_tensor) * self.gamma ** self.n_step_returns * next_q

            loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        self.update_target_counter += 1

        if self.update_target_counter >= self.config.update_target_freq:
            self.update_target_model()
            self.update_target_counter = 0
            print(f"Target model updated. Train step: {self.train_step}, Epsilon: {self.epsilon:.4f}")

        return loss.item()

    def save(self, name="pacman_advanced_dqn_model.pth"):
        self.model.to("cpu")
        self.target_model.to("cpu")
        try:
            directory = os.path.dirname(name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_step': self.train_step,
                'epsilon': self.epsilon,
                'config': vars(self.config)
            }

            torch.save(save_dict, name)
            print(f"Model saved successfully to: {name}")

        except Exception as e:
            print(f"Error saving model: {str(e)}")

        self.model.to(self.device)
        self.target_model.to(self.device)

    def load(self, name="pacman_advanced_dqn_model.pth"):
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

            if 'target_model_state_dict' in checkpoint:
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'], strict=False)
                if hasattr(self.target_model, 'combined_size') and self.target_model.combined_size is None:
                    dummy_matrix = torch.zeros((1, self.state_matrix_shape[2],
                                                self.state_matrix_shape[0],
                                                self.state_matrix_shape[1])).to(self.device)
                    dummy_features = torch.zeros((1, self.state_feature_size)).to(self.device)
                    _ = self.target_model(dummy_matrix, dummy_features)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Could not load optimizer state - initializing new optimizer")

            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']

            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                for key, value in config_dict.items():
                    setattr(self.config, key, value)

            print(f"Model loaded successfully from: {name}")
            print(f"Current epsilon: {self.epsilon}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # traceback.print_exc()  # Print the full traceback for debugging
            return False


def play_pacman(model_path="models/pacman_dqn_best.pth", fps=30):
    env = AdvancedPacmanEnv(render=True, fps=fps)
    initial_state = env.reset()
    state_matrix_shape = initial_state["matrix"].shape
    state_feature_size = len(initial_state["features"])
    config = DQNConfig()
    agent = AdvancedDQNAgent(
        state_matrix_shape=state_matrix_shape,
        state_feature_size=state_feature_size,
        action_size=4,  # UP, RIGHT, DOWN, LEFT
        config=config
    )
    if not agent.load(model_path):
        print(f"Failed to load model from {model_path}")
        env.close()
        return
    agent.epsilon = 0.05
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        action = agent.get_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        print(f"Score: {info['score']}, Lives: {info['lives']}, Reward: {reward:.1f}, Total Reward: {total_reward:.1f}")

    print(f"Game Over! Final Score: {info['score']}")
    time.sleep(2)
    env.close()


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
        # Train agent
        history = train_pacman_dqn(
            episodes=args.episodes,
            render=args.render,
            load_model=args.load_model,
            model_path=args.model_path,
            fps=args.fps
        )

    elif args.mode == "play":
        play_pacman(model_path=args.model_path, fps=args.fps)
