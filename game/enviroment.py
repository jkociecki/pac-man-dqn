import numpy as np
import pygame

from game.game_core import Direction, CELL_SIZE, Ghost, Bonus, Maze, Pacman, FPS
from game.main import GameState, MAZE_LAYOUT


class SimplePacmanEnv:
    def __init__(self, render=True, fps=FPS):
        self.render_enabled = render
        self.fps = fps
        self.screen_width = len(MAZE_LAYOUT[0]) * CELL_SIZE
        self.screen_height = len(MAZE_LAYOUT) * CELL_SIZE

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Simple Pac-Man")
            self.font = pygame.font.SysFont(None, 36)
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        self.reset()

    def reset(self):
        self.maze = Maze(MAZE_LAYOUT)
        pacman_start_x = self.screen_width // 2
        pacman_start_y = self.screen_height // 4 - CELL_SIZE
        self.pacman = Pacman(pacman_start_x, pacman_start_y)

        self.ghosts = [
            Ghost(self.screen_width // 4, self.screen_height * 3 // 4, (255, 0, 0)),
            Ghost(self.screen_width * 3 // 4, self.screen_height * 3 // 4, (0, 255, 0))
        ]

        self.bonuses = [
            Bonus(self.screen_width // 2, self.screen_height // 2)
        ]

        self.game_over = False
        self.level_completed = False
        self.steps_without_reward = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.previous_dots = sum(1 for _, _, active in self.maze.dots if active)

        game_state = GameState(self.pacman, self.ghosts, self.maze, self.bonuses)
        return game_state.get_observation()

    def step(self, action):
        dots_before = sum(1 for _, _, active in self.maze.dots if active)
        power_dots_before = sum(1 for _, _, active in self.maze.power_dots if active)

        self.pacman.next_direction = action
        self.pacman.update(self.maze)
        self.maze.check_dot_collision(self.pacman)

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
                        self.pacman.x = self.screen_width // 2
                        self.pacman.y = self.screen_height // 4 - CELL_SIZE
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

        reward = self.calculate_reward(
            ghost_collision=ghost_collision,
            ghost_eaten=ghost_eaten,
            dots_eaten=dots_eaten,
            power_dots_eaten=power_dots_eaten,
            bonus_eaten=bonus_eaten
        )

        self.episode_reward += reward

        if reward > 0:
            self.steps_without_reward = 0
        else:
            self.steps_without_reward += 1

        self.total_steps += 1

        done = self.game_over or self.level_completed or self.steps_without_reward > 200

        if self.render_enabled:
            self._render()

        game_state = GameState(self.pacman, self.ghosts, self.maze, self.bonuses)
        new_observation = game_state.get_observation()
        info = {
            "score": self.pacman.score,
            "lives": self.pacman.lives,
            "steps": self.total_steps
        }

        return new_observation, reward, done, info

    def calculate_reward(self, ghost_collision, ghost_eaten, dots_eaten, power_dots_eaten, bonus_eaten):
        reward = 0

        reward += dots_eaten * 10
        reward += power_dots_eaten * 50
        reward += ghost_eaten * 100
        reward += bonus_eaten * 25

        if ghost_collision:
            reward -= 100

        if self.level_completed:
            reward += 500
        if self.game_over:
            reward -= 200

        if self.steps_without_reward > 50:
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

        pygame.display.flip()

        if self.clock:
            self.clock.tick(self.fps)

    def close(self):
        if self.render_enabled:
            pygame.quit()
