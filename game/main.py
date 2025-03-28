from game.game_core import *

pygame.init()

MAZE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 3, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class GameState:
    def __init__(self, pacman, ghosts, maze, bonuses):
        self.pacman = pacman
        self.ghosts = ghosts
        self.maze = maze
        self.bonuses = bonuses
        self.score = pacman.score
        self.lives = pacman.lives
        self.game_over = False
        self.level_completed = False

        self.grid_width = maze.width
        self.grid_height = maze.height

        self._update_matrix_representation()

    def _update_matrix_representation(self):

        self.wall_matrix = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.dot_matrix = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.power_dot_matrix = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.pacman_matrix = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.ghost_matrix = np.zeros((self.grid_height, self.grid_width, len(self.ghosts)), dtype=np.int8)
        self.ghost_scared_matrix = np.zeros((self.grid_height, self.grid_width, len(self.ghosts)), dtype=np.int8)
        self.bonus_matrix = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.maze.layout[y][x] == 1:
                    self.wall_matrix[y, x] = 1

        for dot_x, dot_y, active in self.maze.power_dots:
            if active:
                grid_x, grid_y = int(dot_x // CELL_SIZE), int(dot_y // CELL_SIZE)
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    self.power_dot_matrix[grid_y, grid_x] = 1

        pacman_grid_x, pacman_grid_y = int(self.pacman.x // CELL_SIZE), int(self.pacman.y // CELL_SIZE)
        if 0 <= pacman_grid_x < self.grid_width and 0 <= pacman_grid_y < self.grid_height:
            self.pacman_matrix[pacman_grid_y, pacman_grid_x] = 1

        for i, ghost in enumerate(self.ghosts):
            ghost_grid_x, ghost_grid_y = int(ghost.x // CELL_SIZE), int(ghost.y // CELL_SIZE)
            if 0 <= ghost_grid_x < self.grid_width and 0 <= ghost_grid_y < self.grid_height:
                self.ghost_matrix[ghost_grid_y, ghost_grid_x, i] = 1
                if ghost.is_scared:
                    self.ghost_scared_matrix[ghost_grid_y, ghost_grid_x, i] = 1

        for bonus in self.bonuses:
            if bonus.active:
                bonus_grid_x, bonus_grid_y = int(bonus.x // CELL_SIZE), int(bonus.y // CELL_SIZE)
                if 0 <= bonus_grid_x < self.grid_width and 0 <= bonus_grid_y < self.grid_height:
                    if bonus.type == "cherry":
                        self.bonus_matrix[bonus_grid_y, bonus_grid_x] = 1
                    elif bonus.type == "pill":
                        self.bonus_matrix[bonus_grid_y, bonus_grid_x] = 2

    def get_state_features(self):
        self._update_matrix_representation()

        features = [self.pacman.x / (self.grid_width * CELL_SIZE), self.pacman.y / (self.grid_height * CELL_SIZE)]

        direction_one_hot = [0, 0, 0, 0, 0]
        direction_one_hot[self.pacman.direction.value] = 1
        features.extend(direction_one_hot)

        for i, ghost in enumerate(self.ghosts):
            features.append(ghost.x / (self.grid_width * CELL_SIZE))
            features.append(ghost.y / (self.grid_height * CELL_SIZE))

            dx = (ghost.x - self.pacman.x) / (self.grid_width * CELL_SIZE)
            dy = (ghost.y - self.pacman.y) / (self.grid_height * CELL_SIZE)
            distance = np.sqrt(dx ** 2 + dy ** 2)
            features.append(dx)
            features.append(dy)
            features.append(distance)

            features.append(1 if ghost.is_scared else 0)

        features.append(1 if self.pacman.is_powered_up else 0)
        features.append(self.pacman.power_up_timer / (FPS * 10) if self.pacman.is_powered_up else 0)

        dots_remaining = np.sum(self.dot_matrix)
        power_dots_remaining = np.sum(self.power_dot_matrix)
        features.append(dots_remaining / len(self.maze.dots) if len(self.maze.dots) > 0 else 0)
        features.append(power_dots_remaining / len(self.maze.power_dots) if len(self.maze.power_dots) > 0 else 0)

        closest_dot_distance = float('inf')
        closest_power_dot_distance = float('inf')

        for dot_x, dot_y, active in self.maze.power_dots:
            if active:
                dx = dot_x - self.pacman.x
                dy = dot_y - self.pacman.y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < closest_power_dot_distance:
                    closest_power_dot_distance = distance

        features.append(
            closest_dot_distance / (self.grid_width * CELL_SIZE) if closest_dot_distance != float('inf') else 1.0)
        features.append(
            closest_power_dot_distance / (self.grid_width * CELL_SIZE) if closest_power_dot_distance != float(
                'inf') else 1.0)

        return np.array(features)

    def get_matrix_representation(self):

        self._update_matrix_representation()

        channels = 5 + 2 * len(self.ghosts)
        state_matrix = np.zeros((self.grid_height, self.grid_width, channels), dtype=np.float32)

        state_matrix[:, :, 0] = self.wall_matrix
        state_matrix[:, :, 1] = self.dot_matrix
        state_matrix[:, :, 2] = self.power_dot_matrix
        state_matrix[:, :, 3] = self.pacman_matrix
        state_matrix[:, :, 4] = self.bonus_matrix

        for i in range(len(self.ghosts)):
            state_matrix[:, :, 5 + i * 2] = self.ghost_matrix[:, :, i]
            state_matrix[:, :, 6 + i * 2] = self.ghost_scared_matrix[:, :, i]

        return state_matrix

    def get_observation(self):
        return {
            "matrix": self.get_matrix_representation(),
            "features": self.get_state_features(),
            "pacman_position": (self.pacman.x, self.pacman.y),
            "pacman_direction": self.pacman.direction,
            "ghost_positions": [(ghost.x, ghost.y) for ghost in self.ghosts],
            "ghost_scared": [ghost.is_scared for ghost in self.ghosts],
            "score": self.score,
            "lives": self.lives,
            "is_powered_up": self.pacman.is_powered_up,
            "power_up_timer": self.pacman.power_up_timer
        }