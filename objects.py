# pylint: disable=missing-docstring


import random
import math

import pygame

import numpy as np

import gym



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
        if not (dir == self.dir*-1):
            self.dir = dir
    
    def update(self):
        if self.dir == -1:
            self.move_x(1)
        if self.dir == 1:
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

    def is_touching_wall(self, max_x, max_y):
        touched_wall = False
        if self.pos_x < 0:
            self.pos_x = 0
            touched_wall = True
        if self.pos_x >= max_x:
            self.pos_x = max_x
            touched_wall = True

        if self.pos_y <= 0:
            self.pos_y = 0
            touched_wall = True
        if self.pos_y >= max_y:
            self.pos_y = max_y
            touched_wall = True
            
        return touched_wall
    
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


class Food():
    eaten = False
    
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.pos_x = np.random.choice(self.grid_x)
        self.pos_y = np.random.choice(self.grid_y)
        self.eaten = False

    def respawn(self):
        self.pos_x = np.random.choice(self.grid_x)
        self.pos_y = np.random.choice(self.grid_y)
        self.eaten = False


