# Pac-Man AI with Hybrid DQN
AI agent playing Pac-Man using Reinforcement Learning

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

## Gameplay Examples
one of the training episodes
![game/gif.gif](game%2Fgif.gif)
## Usage

The system supports three modes:
- Training mode: `python pacman_dqn.py --mode train`
- AI play mode: `python pacman_dqn.py --mode play`
- Human control mode: `python pacman_dqn.py --mode human`
