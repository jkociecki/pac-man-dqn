import sys
from game_core import *

pygame.init()


class Pacman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = Direction.NONE
        self.next_direction = Direction.NONE
        self.speed = PACMAN_SPEED
        self.radius = CELL_SIZE // 2 - 2
        self.score = 0
        self.lives = 1
        self.is_powered_up = False
        self.power_up_timer = 0

    def update(self, maze):
        if self.next_direction != self.direction and self.next_direction != Direction.NONE:
            test_x, test_y = self.x, self.y

            if self.next_direction == Direction.UP:
                test_y -= self.speed
            elif self.next_direction == Direction.RIGHT:
                test_x += self.speed
            elif self.next_direction == Direction.DOWN:
                test_y += self.speed
            elif self.next_direction == Direction.LEFT:
                test_x -= self.speed

            if not maze.is_wall(test_x, test_y, self.radius):
                self.direction = self.next_direction

        if self.direction == Direction.UP:
            new_y = self.y - self.speed
            if not maze.is_wall(self.x, new_y, self.radius):
                self.y = new_y
        elif self.direction == Direction.RIGHT:
            new_x = self.x + self.speed
            if not maze.is_wall(new_x, self.y, self.radius):
                self.x = new_x
        elif self.direction == Direction.DOWN:
            new_y = self.y + self.speed
            if not maze.is_wall(self.x, new_y, self.radius):
                self.y = new_y
        elif self.direction == Direction.LEFT:
            new_x = self.x - self.speed
            if not maze.is_wall(new_x, self.y, self.radius):
                self.x = new_x

        if self.x < 0:
            self.x = maze.width * CELL_SIZE
        elif self.x > maze.width * CELL_SIZE:
            self.x = 0

        if self.is_powered_up:
            self.power_up_timer -= 1
            if self.power_up_timer <= 0:
                self.is_powered_up = False

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)

        if self.direction == Direction.RIGHT:
            start_angle = 45
            end_angle = 315
        elif self.direction == Direction.LEFT:
            start_angle = 135
            end_angle = 405
        elif self.direction == Direction.UP:
            start_angle = 225
            end_angle = 495
        elif self.direction == Direction.DOWN:
            start_angle = -45
            end_angle = 225
        else:
            start_angle = 0
            end_angle = 360

        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)

        pygame.draw.arc(screen, YELLOW, (self.x - self.radius, self.y - self.radius,
                                         self.radius * 2, self.radius * 2),
                        start_rad, end_rad, self.radius)

        if self.direction != Direction.NONE:
            pygame.draw.polygon(screen, BLACK, [
                (self.x, self.y),
                (self.x + self.radius * np.cos(start_rad), self.y - self.radius * np.sin(start_rad)),
                (self.x + self.radius * np.cos(end_rad), self.y - self.radius * np.sin(end_rad))
            ])


# Klasa Duch
class Ghost:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.direction = random.choice([Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT])
        self.speed = GHOST_SPEED + 1
        self.radius = CELL_SIZE // 2 - 2
        self.is_scared = False
        self.change_direction_counter = 0

        if color == RED:
            self.personality = 0.95
            self.strategy = "direct"
        elif color == PINK:
            self.personality = 0.85
            self.strategy = "direct"
        elif color == ORANGE:
            self.personality = 0.75
            self.strategy = "direct"
        else:
            self.personality = 0.65
            self.strategy = "direct"

    def update(self, maze, pacman):
        self.change_direction_counter -= 1

        if self.change_direction_counter <= 0 or self.is_blocked(maze):
            self.choose_direction(maze, pacman)
            self.change_direction_counter = random.randint(10, 30)  # Skrócony czas reakcji

        new_x, new_y = self.x, self.y

        if self.direction == Direction.UP:
            new_y -= self.speed
        elif self.direction == Direction.RIGHT:
            new_x += self.speed
        elif self.direction == Direction.DOWN:
            new_y += self.speed
        elif self.direction == Direction.LEFT:
            new_x -= self.speed

        if not maze.is_wall(new_x, new_y, self.radius):
            self.x, self.y = new_x, new_y
        else:
            self.choose_direction(maze, pacman)

        if self.x < 0:
            self.x = maze.width * CELL_SIZE
        elif self.x > maze.width * CELL_SIZE:
            self.x = 0

    def is_blocked(self, maze):
        if self.direction == Direction.NONE:
            return True

        test_x, test_y = self.x, self.y

        if self.direction == Direction.UP:
            test_y -= self.speed
        elif self.direction == Direction.RIGHT:
            test_x += self.speed
        elif self.direction == Direction.DOWN:
            test_y += self.speed
        elif self.direction == Direction.LEFT:
            test_x -= self.speed

        return maze.is_wall(test_x, test_y, self.radius)

    def choose_direction(self, maze, pacman):
        possible_directions = []
        direction_offsets = [
            (Direction.UP, 0, -self.speed),
            (Direction.RIGHT, self.speed, 0),
            (Direction.DOWN, 0, self.speed),
            (Direction.LEFT, -self.speed, 0)
        ]

        for direction, dx, dy in direction_offsets:
            if not maze.is_wall(self.x + dx, self.y + dy, self.radius):
                possible_directions.append(direction)

        if not possible_directions:
            opposite_directions = {
                Direction.UP: Direction.DOWN,
                Direction.RIGHT: Direction.LEFT,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT
            }
            self.direction = opposite_directions.get(self.direction, Direction.NONE)
            return

        opposite_directions = {
            Direction.UP: Direction.DOWN,
            Direction.RIGHT: Direction.LEFT,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT
        }

        if len(possible_directions) > 1 and self.direction != Direction.NONE:
            opposite = opposite_directions.get(self.direction)
            if opposite in possible_directions:
                possible_directions.remove(opposite)

        if self.is_scared:
            self.direction = self.choose_direction_away_from_pacman(possible_directions, pacman)
        else:
            if random.random() < self.personality:
                self.direction = self.choose_direction_towards_pacman(possible_directions, pacman)
            else:
                self.direction = random.choice(possible_directions)

    def choose_direction_towards_pacman(self, possible_directions, pacman):
        best_direction = random.choice(possible_directions)
        min_distance = float('inf')

        for direction in possible_directions:
            test_x, test_y = self.x, self.y

            if direction == Direction.UP:
                test_y -= self.speed * 5
            elif direction == Direction.RIGHT:
                test_x += self.speed * 5
            elif direction == Direction.DOWN:
                test_y += self.speed * 5
            elif direction == Direction.LEFT:
                test_x -= self.speed * 5

            distance = (test_x - pacman.x) ** 2 + (test_y - pacman.y) ** 2

            if distance < min_distance:
                min_distance = distance
                best_direction = direction

        return best_direction

    def choose_direction_away_from_pacman(self, possible_directions, pacman):
        best_direction = random.choice(possible_directions)
        max_distance = -1

        for direction in possible_directions:
            test_x, test_y = self.x, self.y

            if direction == Direction.UP:
                test_y -= self.speed * 5
            elif direction == Direction.RIGHT:
                test_x += self.speed * 5
            elif direction == Direction.DOWN:
                test_y += self.speed * 5
            elif direction == Direction.LEFT:
                test_x -= self.speed * 5

            distance = (test_x - pacman.x) ** 2 + (test_y - pacman.y) ** 2

            if distance > max_distance:
                max_distance = distance
                best_direction = direction

        return best_direction

    def draw(self, screen):
        if self.is_scared:
            color = (0, 0, 255)
        else:
            color = self.color

        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)

        points = [(self.x - self.radius, self.y)]
        segment_width = self.radius * 2 // 3
        for i in range(3):
            points.append((self.x - self.radius + segment_width * i, self.y + self.radius))
            points.append((self.x - self.radius + segment_width * (i + 0.5), self.y + self.radius // 2))
        points.append((self.x + self.radius, self.y))

        pygame.draw.polygon(screen, color, points)

        eye_radius = self.radius // 3
        pygame.draw.circle(screen, WHITE, (self.x - eye_radius, self.y - eye_radius), eye_radius)
        pygame.draw.circle(screen, WHITE, (self.x + eye_radius, self.y - eye_radius), eye_radius)

        pupil_radius = eye_radius // 2
        if self.direction == Direction.UP:
            pygame.draw.circle(screen, BLACK, (self.x - eye_radius, self.y - eye_radius - pupil_radius), pupil_radius)
            pygame.draw.circle(screen, BLACK, (self.x + eye_radius, self.y - eye_radius - pupil_radius), pupil_radius)
        elif self.direction == Direction.RIGHT:
            pygame.draw.circle(screen, BLACK, (self.x - eye_radius + pupil_radius, self.y - eye_radius), pupil_radius)
            pygame.draw.circle(screen, BLACK, (self.x + eye_radius + pupil_radius, self.y - eye_radius), pupil_radius)
        elif self.direction == Direction.DOWN:
            pygame.draw.circle(screen, BLACK, (self.x - eye_radius, self.y - eye_radius + pupil_radius), pupil_radius)
            pygame.draw.circle(screen, BLACK, (self.x + eye_radius, self.y - eye_radius + pupil_radius), pupil_radius)
        elif self.direction == Direction.LEFT:
            pygame.draw.circle(screen, BLACK, (self.x - eye_radius - pupil_radius, self.y - eye_radius), pupil_radius)
            pygame.draw.circle(screen, BLACK, (self.x + eye_radius - pupil_radius, self.y - eye_radius), pupil_radius)
        else:
            pygame.draw.circle(screen, BLACK, (self.x - eye_radius, self.y - eye_radius), pupil_radius)
            pygame.draw.circle(screen, BLACK, (self.x + eye_radius, self.y - eye_radius), pupil_radius)


class Bonus:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = CELL_SIZE // 3
        self.active = True
        self.type = "cherry"  # Domyślnie wiśnie

    def draw(self, screen):
        if not self.active:
            return

        if self.type == "cherry":
            pygame.draw.circle(screen, CHERRY_RED, (self.x - 5, self.y + 5), self.radius // 2)
            pygame.draw.circle(screen, CHERRY_RED, (self.x + 5, self.y + 5), self.radius // 2)
            pygame.draw.line(screen, GREEN, (self.x - 5, self.y + 5 - self.radius // 2),
                             (self.x, self.y - 10), 2)
            pygame.draw.line(screen, GREEN, (self.x + 5, self.y + 5 - self.radius // 2),
                             (self.x, self.y - 10), 2)
        elif self.type == "pill":
            pygame.draw.rect(screen, WHITE, (self.x - 8, self.y - 4, 16, 8), border_radius=4)


class Maze:
    def __init__(self, layout):
        self.layout = layout
        self.height = len(layout)
        self.width = len(layout[0])
        self.dots = []
        self.power_dots = []
        self.wall_rects = []

        for y in range(self.height):
            for x in range(self.width):
                if layout[y][x] == 1:  # Ściana
                    self.wall_rects.append(pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif layout[y][x] == 0:  # Kropka
                    self.dots.append((x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2, True))
                elif layout[y][x] == 2:  # Power dot
                    self.power_dots.append((x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2, True))

    def is_wall(self, x, y, radius):
        player_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        for wall_rect in self.wall_rects:
            if player_rect.colliderect(wall_rect):
                return True
        return False

    def draw(self, screen):
        for wall_rect in self.wall_rects:
            pygame.draw.rect(screen, BLUE, wall_rect)
            pygame.draw.rect(screen, BLACK, wall_rect.inflate(-4, -4))

        for dot in self.dots:
            x, y, active = dot
            if active:
                pygame.draw.circle(screen, WHITE, (x, y), DOT_SIZE)

        for power_dot in self.power_dots:
            x, y, active = power_dot
            if active:
                pygame.draw.circle(screen, WHITE, (x, y), POWER_DOT_SIZE)

    def check_dot_collision(self, pacman):
        for i, dot in enumerate(self.dots):
            x, y, active = dot
            if active:
                distance = np.sqrt((pacman.x - x) ** 2 + (pacman.y - y) ** 2)
                if distance < pacman.radius + DOT_SIZE:
                    self.dots[i] = (x, y, False)
                    pacman.score += 10

        for i, power_dot in enumerate(self.power_dots):
            x, y, active = power_dot
            if active:
                distance = np.sqrt((pacman.x - x) ** 2 + (pacman.y - y) ** 2)
                if distance < pacman.radius + POWER_DOT_SIZE:
                    self.power_dots[i] = (x, y, False)
                    pacman.score += 50
                    pacman.is_powered_up = True
                    pacman.power_up_timer = FPS * 10

    def all_dots_eaten(self):
        all_dots_eaten = True
        for _, _, active in self.dots:
            if active:
                all_dots_eaten = False
                break

        all_power_dots_eaten = True
        for _, _, active in self.power_dots:
            if active:
                all_power_dots_eaten = False
                break

        return all_dots_eaten and all_power_dots_eaten


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

import numpy as np


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

        for dot_x, dot_y, active in self.maze.dots:
            if active:
                grid_x, grid_y = int(dot_x // CELL_SIZE), int(dot_y // CELL_SIZE)
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    self.dot_matrix[grid_y, grid_x] = 1

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

        features = []

        features.append(self.pacman.x / (self.grid_width * CELL_SIZE))
        features.append(self.pacman.y / (self.grid_height * CELL_SIZE))

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

        for dot_x, dot_y, active in self.maze.dots:
            if active:
                dx = dot_x - self.pacman.x
                dy = dot_y - self.pacman.y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < closest_dot_distance:
                    closest_dot_distance = distance

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

    def get_legal_actions(self):
        legal_actions = []

        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            test_x, test_y = self.pacman.x, self.pacman.y

            if direction == Direction.UP:
                test_y -= self.pacman.speed
            elif direction == Direction.RIGHT:
                test_x += self.pacman.speed
            elif direction == Direction.DOWN:
                test_y += self.pacman.speed
            elif direction == Direction.LEFT:
                test_x -= self.pacman.speed

            if not self.maze.is_wall(test_x, test_y, self.pacman.radius):
                legal_actions.append(direction)

        return legal_actions

    def calculate_danger_map(self):
        danger_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        for ghost in self.ghosts:
            if ghost.is_scared:
                continue

            ghost_grid_x, ghost_grid_y = int(ghost.x // CELL_SIZE), int(ghost.y // CELL_SIZE)

            danger_radius = 5

            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    if self.wall_matrix[y, x] == 1:
                        continue

                    distance = np.sqrt((x - ghost_grid_x) ** 2 + (y - ghost_grid_y) ** 2)

                    if distance < danger_radius:
                        danger_value = np.exp(-distance)
                        danger_map[y, x] = max(danger_map[y, x], danger_value)

        return danger_map

    def calculate_reward_map(self):
        reward_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        dot_reward = 0.1
        power_dot_reward = 0.5
        bonus_reward = 0.8

        reward_map += self.dot_matrix * dot_reward

        reward_map += self.power_dot_matrix * power_dot_reward

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.bonus_matrix[y, x] > 0:
                    reward_map[y, x] = bonus_reward

        if self.pacman.is_powered_up:
            for i, ghost in enumerate(self.ghosts):
                if ghost.is_scared:
                    ghost_grid_x, ghost_grid_y = int(ghost.x // CELL_SIZE), int(ghost.y // CELL_SIZE)
                    if 0 <= ghost_grid_x < self.grid_width and 0 <= ghost_grid_y < self.grid_height:
                        reward_map[ghost_grid_y, ghost_grid_x] = 0.7

        return reward_map


class PacmanAI:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.action_space = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

    def get_action(self, state_features):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        pacman_x = state_features[0]
        pacman_y = state_features[1]
        pacman_direction = np.argmax(state_features[2:7])

        available_actions = []
        for i, action_available in enumerate(state_features[-5:-1]):
            if action_available > 0.5:  # Jeśli nie ma ściany
                available_actions.append(self.action_space[i])

        if not available_actions:
            return Direction.NONE

        if state_features[-7] > 0.5:
            ghost_distances = []
            for i in range(4):
                ghost_idx = 7 + i * 4
                ghost_distance = state_features[ghost_idx + 2]
                ghost_scared = state_features[ghost_idx + 3]
                if ghost_scared > 0.5:
                    ghost_distances.append((ghost_distance, i))

            if ghost_distances:
                ghost_distances.sort()
                closest_ghost_idx = ghost_distances[0][1]
                ghost_idx = 7 + closest_ghost_idx * 4
                ghost_dx = state_features[ghost_idx]
                ghost_dy = state_features[ghost_idx + 1]

                if abs(ghost_dx) > abs(ghost_dy):
                    if ghost_dx > 0 and Direction.RIGHT in available_actions:
                        return Direction.RIGHT
                    elif ghost_dx < 0 and Direction.LEFT in available_actions:
                        return Direction.LEFT
                else:
                    if ghost_dy > 0 and Direction.DOWN in available_actions:
                        return Direction.DOWN
                    elif ghost_dy < 0 and Direction.UP in available_actions:
                        return Direction.UP

        for i in range(4):
            ghost_idx = 7 + i * 4
            ghost_distance = state_features[ghost_idx + 2]
            ghost_scared = state_features[ghost_idx + 3]

            if ghost_distance < 0.2 and not ghost_scared > 0.5:
                ghost_dx = state_features[ghost_idx]
                ghost_dy = state_features[ghost_idx + 1]

                if abs(ghost_dx) > abs(ghost_dy):
                    if ghost_dx > 0 and Direction.LEFT in available_actions:
                        return Direction.LEFT
                    elif ghost_dx < 0 and Direction.RIGHT in available_actions:
                        return Direction.RIGHT
                else:
                    if ghost_dy > 0 and Direction.UP in available_actions:
                        return Direction.UP
                    elif ghost_dy < 0 and Direction.DOWN in available_actions:
                        return Direction.DOWN

        dot_distance = state_features[-8]
        power_dot_distance = state_features[-7]

        if power_dot_distance < dot_distance:

            return np.random.choice(available_actions)
        else:
            # Idź w kierunku zwykłej kropki
            return np.random.choice(available_actions)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    SCREEN_WIDTH = len(MAZE_LAYOUT[0]) * CELL_SIZE
    SCREEN_HEIGHT = len(MAZE_LAYOUT) * CELL_SIZE

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pac-Man AI")

    maze = Maze(MAZE_LAYOUT)

    pacman_start_x = SCREEN_WIDTH // 2
    pacman_start_y = SCREEN_HEIGHT // 4 - CELL_SIZE
    pacman = Pacman(pacman_start_x, pacman_start_y)

    ghosts = [
        Ghost(SCREEN_WIDTH // 4 + CELL_SIZE, SCREEN_HEIGHT * 3 // 4, RED),
        Ghost(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, PINK),
        Ghost(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4, ORANGE),
        Ghost(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, GREEN)
    ]
    bonuses = [
        Bonus(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    ]

    ai = PacmanAI()

    clock = pygame.time.Clock()

    game_over = False
    game_won = False
    level_completed = False

    bonus_timer = 0
    bonus_active = False

    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_UP:
                    pacman.next_direction = Direction.UP
                elif event.key == pygame.K_RIGHT:
                    pacman.next_direction = Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    pacman.next_direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    pacman.next_direction = Direction.LEFT

        if not game_over and not level_completed:
            game_state = GameState(pacman, ghosts, maze, bonuses)
            state_features = game_state.get_state_features()

            pacman.update(maze)

            maze.check_dot_collision(pacman)

            for ghost in ghosts:
                ghost.is_scared = pacman.is_powered_up
                ghost.update(maze, pacman)

                distance = np.sqrt((ghost.x - pacman.x) ** 2 + (ghost.y - pacman.y) ** 2)
                if distance < (ghost.radius + pacman.radius) * 0.8:
                    if ghost.is_scared:
                        ghost.x = SCREEN_WIDTH // 4 + CELL_SIZE
                        ghost.y = SCREEN_HEIGHT * 3 // 4
                        pacman.score += 200
                    else:
                        pacman.lives -= 1
                        if pacman.lives <= 0:
                            game_over = True
                        else:
                            pacman.x = pacman_start_x
                            pacman.y = pacman_start_y
                            pacman.direction = Direction.NONE
                            pacman.next_direction = Direction.NONE
                            for g in ghosts:
                                g.x = SCREEN_WIDTH // 2
                                g.y = SCREEN_HEIGHT // 2

            for bonus in bonuses:
                if bonus.active:
                    distance = np.sqrt((bonus.x - pacman.x) ** 2 + (bonus.y - pacman.y) ** 2)
                    if distance < (bonus.radius + pacman.radius):
                        bonus.active = False
                        pacman.score += 100
                        if bonus.type == "cherry":
                            pacman.score += 100
                        elif bonus.type == "pill":
                            pacman.lives += 1

            bonus_timer += 1
            if bonus_timer > FPS * 30:
                bonus_timer = 0
                for bonus in bonuses:
                    if not bonus.active:
                        bonus.active = True
                        bonus.type = random.choice(["cherry", "pill"])
                        break

            if maze.all_dots_eaten():
                level_completed = True

            ai.update_epsilon()

        screen.fill(BLACK)

        maze.draw(screen)

        for bonus in bonuses:
            bonus.draw(screen)

        for ghost in ghosts:
            ghost.draw(screen)

        pacman.draw(screen)
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        lives_text = font.render(f"Lives: {pacman.lives}", True, WHITE)
        screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))

        if game_over:
            game_over_text = font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(game_over_text, text_rect)

            restart_text = font.render("Press ESC to quit", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            screen.blit(restart_text, restart_rect)

        if level_completed:
            level_text = font.render("LEVEL COMPLETED!", True, GREEN)
            text_rect = level_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(level_text, text_rect)

            next_level_text = font.render("Press ESC to quit", True, WHITE)
            next_rect = next_level_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            screen.blit(next_level_text, next_rect)
        pygame.display.flip()

        clock.tick(FPS)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
