from enum import Enum

import numpy as np
import pygame
import random

CELL_SIZE = 30
GHOST_SPEED = 2
PACMAN_SPEED = 5
DOT_SIZE = 4
POWER_DOT_SIZE = 10
FPS = 60

BLACK = (0, 0, 0)
BLUE = (33, 33, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 105, 180)
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
CHERRY_RED = (220, 20, 60)


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4


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
