# Pac-Man AI with Hybrid DQN
AI agent playing Pac-Man using Deep Reinforcement Learning with **Hybrid Neural Network** (CNN + Dense) and prioritized experience replay.

## Game State Representation
The environment is encoded as multi-channel matrix + features:
**Matrix Channels (7 layers):**
1. `Walls` - Binary map
2. `Dots` - Remaining pellets
3. `Power Dots` - Energizers
4. `Pac-Man` - Current position
5-6. `Ghosts` - Positions + scared status
7. `Bonuses` - Cherry/pill locations
**Feature Vector (25+ values):**
- Pac-Man position/direction
- Ghost distances/statuses
- Power-up timer
- Closest dot/pellet info
- Danger/reward map metrics

## Learning Visualization Tool

This is a simple but quite interesting tool for visualizing neural network learning in action. I've experimented with various hyperparameters and reward structures to see how the agent adapts to the game environment.

One of cool aspects was watching how the model eventually learned to chase ghosts after eating power pellets instead of running away from them.

In the current game settings, Pac-Man has a significant handicap with an extended power pellet duration and ghosts that don't move after being eaten. I've played around with these settings though - reducing the power duration creates a much more challenging environment where timing becomes critical. You can easily modify these parameters in the code to create different difficulty levels.

## Advanced Features

- **Prioritized Experience Replay**: Helps the agent learn from important experiences more efficiently
- **Double DQN**: Reduces overestimation of Q-values
- **Dueling Networks**: Separates value and advantage streams for better policy evaluation
- **N-step Returns**: Balances immediate and future rewards
- **Reward Shaping**: Custom reward function that encourages strategic behavior

## Gameplay Examples
Below are a few recorded gameplay samples showing different stages of training and strategic behaviors. You can see how the agent progresses from random movements to increasingly sophisticated strategies:

![Early training gameplay](https://github.com/user-attachments/assets/48e2ef70-c6cd-4b7a-b8c3-cd5a5f356796)
![Advanced strategy gameplay](https://github.com/user-attachments/assets/b69cf1cc-1ec7-45a6-a5f4-e80fc5f367bc)


## Usage

The system supports three modes:
- Training mode: `python pacman_dqn.py --mode train`
- AI play mode: `python pacman_dqn.py --mode play`
- Human control mode: `python pacman_dqn.py --mode human`
