import pygame
import pygame.freetype
from objects import SnakeWorld

import numpy as np

from qagent import QNet
import torch

SCREEN_WIDTH_IN_SQUARES = 20
SCREEN_HEIGHT_IN_SQUARES = 20

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

def make_states(state):
    state1 = (state == 1).astype(int)
    state2 = (state == 2).astype(int)
    state3 = (state == 3).astype(int)
    return state1, state2, state3


world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)
world.reinitialise()
game_over = False

score = 0
state1, state2, state3 = make_states(world.state)
state = np.asarray([state1, state2, state3])
state = torch.Tensor(state)
state = state.unsqueeze(0)

while not game_over:
    steps += 1
    dir = target_net.get_action(state)
    print(steps, dir)
    world.render()

    next_state, game_over, _, reward = world.step(dir)

    next_state1, next_state2, next_state3 = make_states(next_state)
    next_state = np.asarray([next_state1, next_state2, next_state3])
    next_state = torch.Tensor(next_state)

    next_state = next_state.unsqueeze(0)
    state = next_state



