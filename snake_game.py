import pygame
import pygame.freetype
from objects import SnakeWorld

MAX_LENGTH = 10

SCREEN_WIDTH_IN_SQUARES = 50
SCREEN_HEIGHT_IN_SQUARES = 50

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000


RUNNING = True

world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)

game_over = False

while not game_over:
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
    world.render()
            


