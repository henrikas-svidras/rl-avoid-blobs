# pylint: disable=missing-docstring


import random
import math

import pygame

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define a Player object by extending pygame.sprite.Sprite

# The surface drawn on the screen is now an attribute of 'player'

class Player(pygame.sprite.Sprite):
    def __init__(self, bot=False):
        super(Player, self).__init__()
        w, h = pygame.display.get_surface().get_size()

        self.surf = pygame.Surface((5, 5))
        self.surf.fill((255, 255, 255))
        #init_x = random.randint(0+self.surf.get_width(), w-self.surf.get_width())
        #init_y = random.randint(0+self.surf.get_height(), h-self.surf.get_height())
        init_x = w*0.1
        init_y= h/2
        self.rect = self.surf.get_rect(center=(init_x, init_y))
        self.speed_x = 0
        self.speed_y = 0
        self.max_speed = 10
        self.bot = bot

        self.acceleration_x = 0
        self.acceleration_y = 0

        self.hitting_side_wall = False
        self.hitting_top_wall = False

    def accelerate(self):
        self.speed_x += self.acceleration_x*2.5
        self.speed_y += self.acceleration_y*2.5

    def update(self, pressed_keys=None, acceleration=None):
        w, h = pygame.display.get_surface().get_size()

        if not self.bot:
            if pressed_keys[pygame.K_UP]:
                self.rect.move_ip(0, -self.speed_y)
            if pressed_keys[pygame.K_DOWN]:
                self.rect.move_ip(0, self.speed_y)
            if pressed_keys[pygame.K_LEFT]:
                self.rect.move_ip(-self.speed_x, 0)
            if pressed_keys[pygame.K_RIGHT]:
                self.rect.move_ip(self.speed_x, 0)
        else:

            # goal = decide(enemy_xy, w//2, h//2)
            #self.acceleration_x = acceleration[0]
            #self.acceleration_y = acceleration[1]
            #self.accelerate()
            #self.drag()
            self.rect.move_ip(acceleration[0]*7, acceleration[1]*7)

        self.respect_bounds(w, h)

    def respect_bounds(self, w, h):
        """
        Method that ensures the "Player" can't leave the grid.
        """

        if self.rect.left < 0:
            self.rect.left = 0
            self.speed_x = 0
            self.hitting_side_wall = True
        elif self.rect.right > w:
            self.rect.right = w
            self.speed_x = 0
            self.hitting_side_wall = True
        else:
            self.hitting_side_wall = False

        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y = 0
            self.hitting_top_wall = True
        elif self.rect.bottom >= h:
            self.rect.bottom = h
            self.speed_y = 0
            self.hitting_top_wall = True
        else:
            self.hitting_top_wall = False


    def move_to(self, posx, posy):
        diff_x = posx - self.rect.centerx
        sgn_x = 1 if diff_x > 0 else (-1 if diff_x < 0 else 0)
        diff_y = posy - self.rect.centery
        sgn_y = 1 if diff_y > 0 else (-1 if diff_y < 0 else 0)

        self.speed_x -= sgn_x if abs(self.speed_x) < self.max_speed else self.speed_x
        self.speed_y -= sgn_y if abs(self.speed_y) < self.max_speed else self.speed_y

        self.rect.move_ip(self.speed_x, self.speed_y)

    def drag(self):
        if self.speed_x > 0:
            self.speed_x -= 0.1
        if self.speed_y > 0:
            self.speed_y -= 0.1

        if self.speed_x < 0:
            self.speed_x += 0.1
        if self.speed_y < 0:
            self.speed_y += 0.1

# Define the enemy object by extending pygame.sprite.Sprite

# The surface you draw on the screen is now an attribute of 'enemy'

class Enemy(pygame.sprite.Sprite):

    def __init__(self):

        super(Enemy, self).__init__()
        w, h = pygame.display.get_surface().get_size()
        self.surf = pygame.Surface((20, 10))
        self.surf.fill((122, 122, 122))
        init_x = w - self.surf.get_width()
        init_y = random.randint(0, h)
        self.rect = self.surf.get_rect(center=(init_x, init_y))
        self.speed_x = random.randint(1, 9)
        self.speed_y = random.randint(1, 9)
        self.max_speed = 9

    # Move the sprite based on speed

    def steer(self, posx, posy):
        diff_x = posx - self.rect.centerx
        sgn_x = 1 if diff_x>0 else (-1 if diff_x<0 else 0)
        diff_y = posy - self.rect.centery
        sgn_y = 1 if diff_y>0 else (-1 if diff_y<0 else 0)

        factor_x = 1/(abs(diff_x)**(1/4)) if diff_x != 0 else 0
        factor_y = 1/(abs(diff_y)**(1/4)) if diff_y != 0 else 0

        self.speed_x -= factor_x*sgn_x*3 if abs(self.speed_x) < self.max_speed else self.speed_x
        self.speed_y -= factor_y*sgn_y*3 if abs(self.speed_y) < self.max_speed else self.speed_y

    def drag(self):
        if self.speed_x > 0:
            self.speed_x -= 0.1
        if self.speed_y > 0:
            self.speed_y -= 0.1

        if self.speed_x < 0:
            self.speed_x += 0.1
        if self.speed_y < 0:
            self.speed_y += 0.1

    def update(self):
        w, h = pygame.display.get_surface().get_size()
        self.rect.move_ip(-self.speed_x, -self.speed_y)

        if self.rect.left < 0:
            self.rect.left = 0
            self.speed_x*=-1
            if random.random()>0.5:
                self.speed_y += random.uniform(-5,5) if abs(self.speed_y)<self.max_speed else self.max_speed

        if self.rect.right > w:
            self.rect.right = w
            self.speed_x*=-1
            if random.random()>0.5:
                self.speed_y += random.uniform(-5,5) if abs(self.speed_y)<self.max_speed else self.max_speed

        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y*=-1
            if random.random()>0.5:
                self.speed_x += random.uniform(-5,5) if abs(self.speed_x)<self.max_speed else self.max_speed

        if self.rect.bottom >= h:
            self.rect.bottom = h
            self.speed_y*=-1
            if random.random()>0.5:
                self.speed_x += random.uniform(-5,5) if abs(self.speed_x)<self.max_speed else self.max_speed


class Snake:
    
    def __init__(self, leader=None, init_x=None, init_y=None):        
        # Order in tail
        self.order = 0 
        # if +/-1 -> facing left/right, if +/-2 facing up/down       
        self.facing = 1

        self.pos_x = init_x if init_x else 0
        self.pos_y = init_y if init_y else 0
        
        # +1: left, -1: right, +2:up, -2:down
        self.dir = 1 

        # if this snake has a leading part it is checked here
        if leader is not None:
            assert isinstance(
                leader, Snake), "Must follow another Snakepiece or be leading!"
        self.leader = leader
        # upon creation snake doesnt have a follower
        self.follower = None

        if self.leader is not None:
            self.order = self.leader.order + 1
    
    def set_dir(self, dir):
        if dir == 0:
            pass
        elif not (dir == self.dir*-1):
            self.dir = dir
    
    def update(self):
        if self.dir == 1:
            self.move_x(1)
        if self.dir == -1:
            self.move_x(-1)
        if self.dir == 2:
            self.move_y(-1)
        if self.dir == -2:
            self.move_y(1)

    def move_x(self, dir=1):
        self.follow_leader()
        self.pos_x += dir
        self.facing = dir*1

    def move_y(self, dir=1):
        self.follow_leader()
        self.pos_y += dir 
        self.facing = dir*2

    def follow_leader(self):
        if self.follower is not None:
            self.follower.follow_leader()
            self.follower.pos_x = self.pos_x
            self.follower.pos_y = self.pos_y

    def is_touching_wall(self, max_x, max_y, min_x=0, min_y=0):
        touched_wall = False
        if self.pos_x < min_x:
            self.pos_x = min_x
            touched_wall = True
        if self.pos_x > max_x:
            self.pos_x = max_x
            touched_wall = True

        if self.pos_y < min_y:
            self.pos_y = min_y
            touched_wall = True
        if self.pos_y > max_y:
            self.pos_y = max_y
            touched_wall = True

        return touched_wall
    
    def is_self_colliding(self):
        for snakepiece in self.return_self_and_followers()[1:]:
            if (self.pos_x == snakepiece.pos_x) and (self.pos_y == snakepiece.pos_y):
                return True
        return False
    
    def grow(self):
        if self.follower is None:

            init_x = self.pos_x
            init_y = self.pos_y


            if abs(self.facing) == 1:
                init_x -= self.facing
            else:
                init_y -= self.facing//2

            self.follower = Snake(init_x=init_x, init_y=init_y)

        else:
            self.follower.grow()

    def return_self_and_followers(self):
        self_and_followers = np.array([self])
        if self.follower is not None:
            self_and_followers = np.append(self_and_followers,
                                           self.follower.return_self_and_followers())
        return self_and_followers


class Food:
    eaten = False
    
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.pos_x = np.random.choice(self.grid_x)
        self.pos_y = np.random.choice(self.grid_y)
        self.eaten = False

    def respawn(self, state = None):
        if state is None:
            self.pos_x = np.random.choice(self.grid_x)
            self.pos_y = np.random.choice(self.grid_y)
        else:
            available_slots = np.where(state==0)
            random_sel = np.random.choice(len(available_slots[0]))
            self.pos_x = available_slots[0][random_sel]
            self.pos_y = available_slots[1][random_sel]
        self.eaten = False


class SnakeWorld:

    snake = None
    food = None
    x_grid_length = 0
    y_grid_length = 0

    state = None
    past_states = []

    score = 0
    game_over = False
    time_after_death = 0

    ready_to_render = False

    def setup_rendering(self, size):
        pygame.init()
        self.screen = pygame.display.set_mode([size, size])
        self.myfont = pygame.freetype.SysFont('Comic Sans MS', 30)
        self.clock = pygame.time.Clock()
        self.ready_to_render = True

    def __init__(self, x_grid_length, y_grid_length):
        self.x_grid_length = x_grid_length
        self.y_grid_length = y_grid_length
        self.reinitialise()
    
    def reinitialise(self):
        self.snake = Snake(init_x=self.x_grid_length//2,
                           init_y=self.y_grid_length//2)
        self.food = Food(grid_x=self.x_grid_length,
                         grid_y=self.y_grid_length)

        self.state = np.zeros((self.x_grid_length-2, self.y_grid_length-2))
        self.state = np.pad(self.state, 1, constant_values=4)

        self.state[self.snake.pos_x, self.snake.pos_y] = 2
        self.food.respawn(self.state)
        self.state[self.food.pos_x, self.food.pos_y] = 3

        self.past_states = []

        self.score = 0
        self.game_over = False
        self.time_after_death = 0
        self.ready_to_render = False
    
    def renew_state(self):
        self.past_states.append(self.state)
        self.state = np.zeros((self.x_grid_length-2, self.y_grid_length-2))
        self.state = np.pad(self.state, 1, constant_values=4)


    def step(self, snake_dir):
        reward = 0
        # Before drawing sets everything in the state matrix to 0:
        # and saves last state
        self.renew_state()

        # At the beginning of each step respawns the food if it was eaten
        if self.food.eaten:
            self.food.respawn()
        # Sets the food as 3 in the state matrix
        self.state[self.food.pos_x, self.food.pos_y] = 3
        
        # Sets the direction of the snake and updates its entire position
        self.snake.set_dir(snake_dir)
        self.snake.update()

        # If after moving the snake goes into itself or the wall -> game over
        self.game_over = self.snake.is_touching_wall(max_x = self.x_grid_length-2, 
                                                     max_y = self.y_grid_length-2,
                                                     min_x = 1,
                                                     min_y = 1) or\
                                                     self.snake.is_self_colliding()
        print(self.game_over,self.snake.pos_x, self.snake.pos_y)
        if self.game_over:
            reward = -1.

        # If after moving the snake eats the food, set the food state to eaten and grow the snake
        if (self.snake.pos_x == self.food.pos_x) and (self.snake.pos_y == self.food.pos_y):
            self.food.eaten = True
            self.snake.grow()
            self.score += 1
            reward = 10.

        # Sets the snakepieces as 1 on the matrix and head as 2
        for snakepiece in self.snake.return_self_and_followers():
            self.state[snakepiece.pos_x, snakepiece.pos_y] = 1
        self.state[self.snake.pos_x, self.snake.pos_y] = 2
        
        return self.state, self.game_over, self.score, reward

    def render(self, size = 1000, fps = 100):
        if not self.ready_to_render:
            self.setup_rendering(size)
        # Drawing
        self.screen.fill((0, 0, 0))
        self.draw_grid(self.state, size=size)
        self.myfont.render_to(self.screen, (size * 0.8,
                                            size * 0.1), f"Score: {self.score}", (220, 220, 220))
        
        if self.game_over:
            self.myfont.render_to(self.screen, (size * 0.45,
                                                size * 0.5), f"GAME OVER!", (220, 220, 220))
            self.time_after_death += self.clock.get_time()/1000
        pygame.display.flip()
        self.clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
    def render_mpl(self, size=10, fps=30):
        self.animation_finished = False
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

        ini_state = self.past_states[0]
        im = plt.imshow(ini_state, cmap='magma')

        anim = FuncAnimation(
            fig,
            self.swap_state_picture,
            fargs=[im],
            frames=len(self.past_states),
            interval=1000/fps,
            repeat=False,
        )

        plt.show(block=False)
        plt.xlabel([])
        plt.ylabel([])

        plt.pause(1)
        plt.close()

    def swap_state_picture(self, i, im):
        im.set_array(self.past_states[i])
        if i==len(self.past_states)-1:
            print('çlosing')
        return [im]

    def draw_grid(self, screen_array, size=1000):
        square_size = size//self.x_grid_length
        for row in range(self.x_grid_length):
            for column in range(self.y_grid_length):
                entry = screen_array[row, column]
                if entry == 0:
                    continue
                if screen_array[row, column] == 1:
                    color = (200,200,200)
                elif screen_array[row, column] ==2:
                    color = (255, 0, 0)
                elif screen_array[row, column] == 3:
                    color = (0, 255, 0)
                elif screen_array[row, column] == 4:
                    color = (122, 255, 53)
                pygame.draw.rect(self.screen,
                                color,
                                    [(square_size) * row,
                                    (square_size) * column,
                                    square_size,
                                    square_size])

    def __del__(self):
        pygame.quit()


    
