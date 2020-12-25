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
