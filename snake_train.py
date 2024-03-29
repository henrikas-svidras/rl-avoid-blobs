import pygame.freetype
from objects import SnakeWorld, get_action

import numpy as np

from qagent import QNet, Memory
import torch
import torch.optim as optim
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
print("Using device", dev)

SCREEN_WIDTH_IN_SQUARES = 10
SCREEN_HEIGHT_IN_SQUARES = 10

# hyper pars
delta_epsilon = 0.00001
batch_size = 32
lr = 0.001
initial_exploration = 33
log_interval = 100
update_target = 100
replay_memory_capacity = 10000
### 

# NN
#num_inputs = env.observation_space.shape[0]
num_inputs = 1200
num_actions = 3
print('state size:', num_inputs)
print('action size:', num_actions)

online_net = QNet(num_inputs, num_actions, dev=dev)
target_net = QNet(num_inputs, num_actions, dev=dev)

online_net.train()
target_net.train()

memory = Memory(capacity=replay_memory_capacity)

running_score = 0
epsilon = 1.0
steps = 0
loss = 0

optimizer = optim.Adam(online_net.parameters(), lr=lr)

steps = 0


def update_target_model(online_net, target_net):
    # Target <- Net
    # Copies over the weights
    target_net.load_state_dict(online_net.state_dict())

###

game_over = False
training = True


world = SnakeWorld(SCREEN_WIDTH_IN_SQUARES, SCREEN_HEIGHT_IN_SQUARES)
for e in range(100000):
    # hyperparameter to balance risk/reward
    epsilon -= delta_epsilon
    epsilon = max(epsilon, 0.1)
    game_over = False

    score = 0
    world.reinitialise()
    state = torch.Tensor(world.state)
    state = state.unsqueeze(0).unsqueeze(0)
    
    while not game_over:
        steps += 1
        state = state.to(dev)
        dir = get_action(state, target_net, epsilon)
        next_state, game_over, _, reward = world.step(dir)

        next_state = torch.Tensor(next_state)
        next_state = next_state.unsqueeze(0).unsqueeze(0)

        mask = 0 if game_over else 1
        reward = reward if not game_over else -1
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

    running_score = 0.99 * running_score + 0.01 * score
    if e % log_interval == 0:
        print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
            e, running_score, epsilon))
    if e % (10*log_interval) == 0:
        model_generation =  int(e / 1000)
        online_net.save(model_generation)
        print("saving model generation", model_generation)
        with open('log.txt', 'a') as f:
            f.write(f"Gen {model_generation}, score: {running_score},  \n")


