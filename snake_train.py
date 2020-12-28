import pygame
import pygame.freetype
from objects import SnakeWorld

import numpy as np

from qagent import QNet, Memory
import torch
import torch.optim as optim
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
print("Using device", dev)

SCREEN_WIDTH_IN_SQUARES = 20
SCREEN_HEIGHT_IN_SQUARES = 20

# hyper pars
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 33
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000000
### 

# NN
#num_inputs = env.observation_space.shape[0]
num_inputs = 1200
num_actions = 5
print('state size:', num_inputs)
print('action size:', num_actions)

online_net = QNet(num_inputs, num_actions, dev=dev)
target_net = QNet(num_inputs, num_actions, dev=dev)

online_net.train()
target_net.train()

memory = Memory(capacity=100000)

running_score = 0
epsilon = 1.0
steps = 0
loss = 0

optimizer = optim.Adam(online_net.parameters(), lr=lr)

steps = 0

def get_action(state, target_net, epsilon):
    choice_space = [-2,-1,0,1,2]
    if np.random.rand() <= epsilon:
        return np.random.choice(choice_space)
    else:
        return choice_space[target_net.get_action(state)]


def update_target_model(online_net, target_net):
    # Target <- Net
    # Copies over the weights
    target_net.load_state_dict(online_net.state_dict())

###

game_over = False
training = True


def make_states(state):
    state1 = (state == 1).astype(int)
    state2 = (state == 2).astype(int)
    state3 = (state == 3).astype(int)
    return state1, state2, state3


world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)
for e in range(100000):
    # hyperparameter to balance risk/reward
    epsilon -= 0.00001
    epsilon = max(epsilon, 0.1)
    game_over = False

    score = 0
    world.reinitialise()
    state1, state2, state3 = make_states(world.state)
    state = np.asarray([state1, state2, state3])
    state = torch.Tensor(state)
    state = state.unsqueeze(0)
    
    while not game_over:
        steps += 1
        state = state.to(dev)
        dir = get_action(state, target_net, epsilon)
        next_state, game_over, _, reward = world.step(dir)

        #if e % 100 == 0:
            #world.render()

        next_state1, next_state2, next_state3 = make_states(next_state)
        next_state = np.asarray([next_state1, next_state2, next_state3])
        next_state = torch.Tensor(next_state)

        next_state = next_state.unsqueeze(0)

        mask = 0 if game_over else 1
        reward = reward if not game_over or score == 499 else -1
        action_one_hot = np.zeros(num_actions)
        action_one_hot[dir] = 1
        memory.push(state, next_state, action_one_hot, reward, mask)

        score += reward
        state = next_state

        if steps > initial_exploration:
            batch = memory.sample(batch_size)
            loss = online_net.train_model(online_net=online_net, target_net=target_net,
                                          optimizer=optimizer, batch=batch)

            if steps % update_target == 0:
                update_target_model(online_net, target_net)

    score = score if score == 500.0 else score + 1
    running_score = 0.99 * running_score + 0.01 * score
    if e % 100 == 0:
        model_generation =  int(e / 100)
        online_net.save(model_generation)
        print('{} episode | score: {:.2f} | epsilon: {:.2f}. Saving model generation {:n}'.format(
            e, running_score, epsilon, model_generation))
        print

    #if e % 100 == 0:
        #world.render_mpl()

    if running_score > goal_score:
        break


