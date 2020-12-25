import pygame
import pygame.freetype
import numpy as np

from objects import Snake, Food

pygame.init()

MAX_LENGTH = 10

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

RUNNING = True

snake = Snake(init_x=np.random.choice(np.linspace(10, SCREEN_WIDTH-10, SCREEN_WIDTH//5+1)), 
              init_y=np.random.choice(np.linspace(10, SCREEN_WIDTH-10, SCREEN_WIDTH//5+1)))
myfont = pygame.freetype.SysFont('Comic Sans MS', 30)
clock = pygame.time.Clock()

food_on_grid = False

while RUNNING:
    
    if not food_on_grid:
        food = Food(np.random.choice(np.linspace(2.5, SCREEN_WIDTH-2.5, SCREEN_WIDTH//5+1)),
                    np.random.choice(np.linspace(2.5, SCREEN_HEIGHT-2.5, SCREEN_WIDTH//5+1)))
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
    if pygame.sprite.collide_rect(snake, food):
        snake.grow()
        food_on_grid = False
        food.kill()
    screen.fill((0, 0, 0))
    screen.blit(food.surf, food.rect)
    for n, snakepiece in enumerate(snake.return_self_and_followers()):
        screen.blit(snakepiece.surf, snakepiece.rect)
    
    snake.respect_bounds(SCREEN_WIDTH, SCREEN_HEIGHT)

    
    
    pygame.display.flip()
    clock.tick(30)
