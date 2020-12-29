import pygame
import pygame.freetype
from objects import SnakeWorld

import numpy as np

from qagent import QNet
import torch

SCREEN_WIDTH_IN_SQUARES = 10
SCREEN_HEIGHT_IN_SQUARES = 10

# NN
#num_inputs = env.observation_space.shape[0]
num_inputs = 1200
num_actions = 5
print('state size:', num_inputs)
print('action size:', num_actions)

target_net = QNet(num_inputs, num_actions)

target_net.load(1)

steps = 0

game_over = False

def get_action(state, target_net, epsilon=0.05):
    choice_space = [-2,-1,0,1,2]
    if np.random.rand() <= epsilon:
        return np.random.choice(choice_space)
    else:
        return choice_space[target_net.get_action(state)]


world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)
#world.reinitialise()
game_over = False

score = 0
state = torch.Tensor(world.state)
state = state.unsqueeze(0).unsqueeze(0)

while not game_over:
    steps += 1
    dir = get_action(state, target_net)

    next_state, game_over, _, reward = world.step(dir)

    next_state = torch.Tensor(next_state)
    next_state = next_state.unsqueeze(0).unsqueeze(0)

    state = next_state

world.render_mpl(size=5)


