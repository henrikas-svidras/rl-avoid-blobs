import numpy as np
import pygame
import pygame.freetype
from objects import Snake, Food, SnakeWorld

pygame.init()


MAX_LENGTH = 10

SCREEN_WIDTH_IN_SQUARES = 50
SCREEN_HEIGHT_IN_SQUARES = 50

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
myfont = pygame.freetype.SysFont('Comic Sans MS', 30)


RUNNING = True

world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)

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
score = 0
game_over = False
TIME_BETWEEN_REFRESH = 1.5
time_after_death = 0



while RUNNING:
    # This is basically one game cycle:
    if not game_over:   
        dir = 0
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                RUNNING = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dir = 1
                elif event.key == pygame.K_RIGHT:
                    dir = -1
                elif event.key == pygame.K_UP:
                    dir = 2
                elif event.key == pygame.K_DOWN:
                    dir = -2
    
        state, game_over, score = world.step(dir)
    # Here the game cycle ends. You now have access to variables:
    # state - 2D numpy array 
    # game_over - has the snake hit a wall or itself

    # Drawing
    screen.fill((0, 0, 0))
    myfont.render_to(screen, (SCREEN_WIDTH*0.8, 
                              SCREEN_HEIGHT * 0.1), f"Score: {score:}", (220, 220, 220))

    if game_over:
        myfont.render_to(screen, (SCREEN_WIDTH*0.5,
                                  SCREEN_HEIGHT * 0.5), f"GAME OVER!", (220, 220, 220))
        time_after_death += clock.get_time()/1000
        if time_after_death > TIME_BETWEEN_REFRESH:
            RUNNING = False

    draw_grid(state)

    pygame.display.flip()
    clock.tick(10)
pygame.quit()

