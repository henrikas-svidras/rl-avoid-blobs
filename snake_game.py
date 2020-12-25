from core import GAME_OVER
import numpy as np
import pygame
from objects import Snake, Food

pygame.init()


MAX_LENGTH = 10

SCREEN_WIDTH_IN_SQUARES = 50
SCREEN_HEIGHT_IN_SQUARES = 50

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])


RUNNING = True

snake = Snake(init_x=np.random.choice(SCREEN_WIDTH_IN_SQUARES), 
              init_y=np.random.choice(SCREEN_HEIGHT_IN_SQUARES))


food_on_grid = False
clock = pygame.time.Clock()


def draw_grid(screen_array):
    square_size = SCREEN_HEIGHT//SCREEN_HEIGHT_IN_SQUARES
    for row in range(SCREEN_WIDTH_IN_SQUARES):
        for column in range(SCREEN_HEIGHT_IN_SQUARES):
            entry = screen_array[row, column]
            if entry == 0:
                continue
            if screen_array[row, column] == 1:
                color = (200,200,200)
            elif screen_array[row, column] ==2:
                color = (0, 255, 0)
            pygame.draw.rect(screen,
                            color,
                                [(square_size) * row,
                                (square_size) * column,
                                square_size,
                                square_size])

while RUNNING:
    
    state = np.zeros(
        (SCREEN_WIDTH_IN_SQUARES,
        SCREEN_HEIGHT_IN_SQUARES)
    )
    if not food_on_grid and not GAME_OVER:
        food = Food(init_x=np.random.choice(SCREEN_WIDTH_IN_SQUARES),
                    init_y=np.random.choice(SCREEN_HEIGHT_IN_SQUARES))
        food_on_grid = True

    for event in pygame.event.get():
        
        if event.type == pygame.QUIT:
            RUNNING = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                snake.move_y(-1)
            elif event.key == pygame.K_DOWN:
                snake.move_y(1)
            elif event.key == pygame.K_LEFT:
                snake.move_x(-1)
            elif event.key == pygame.K_RIGHT:
                snake.move_x(1)

    wall_touch = snake.is_touching_wall(SCREEN_WIDTH_IN_SQUARES-1, SCREEN_HEIGHT_IN_SQUARES-1)

    state[food.pos_x, food.pos_y] = 2
    for snakepiece in snake.return_self_and_followers():
        state[snakepiece.pos_x, snakepiece.pos_y] = 1

    screen.fill((0, 0, 0))
    draw_grid(state)

    pygame.display.flip()
    clock.tick(1000)
pygame.quit()

