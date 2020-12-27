import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from collections import namedtuple, deque

class QAgent():
    # Algorithm parameters: step size alpha, small step epsilon > 0
    #Initialize Q(s, a) for all state-action pairs randomly
    # Note: terminal needs to have Q(s, .) = 0
    def __init__(self, n_states, n_actions):
        self.Q_state = np.zeros((n_states, n_actions))
        print("Initialized Q-table")
        print(self.Q_state)
        self.alpha = 0.1
        self.epsilon = 0.01

    def choose_action(self, state, env):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    # Bellman equation
    # Update Q value
    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += (self.alpha * reward 
                                    + self.epsilon * np.max(self.Q_table[new_state])
                                    - self.Q_table[state][action]
        )

    def learn(self, env, initial_state):
        state = initial_state
        DONE = False
        while not DONE:
            action = self.choose_action(state, env)
            reward, new_state, DONE = env.get(action)
            self.update_q(state, action, reward, new_state)
            state = new_state
        
        print("finished training!")


def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lr = 0.01
        self.gamma = 0.95
        '''
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        '''

        self.model = nn.Sequential(
          nn.Conv2d(3, 6, 3),
          nn.MaxPool2d(2),
          nn.ReLU(),
          nn.Conv2d(6, 12, 2, dilation=2),
          nn.MaxPool2d(2),
          nn.Conv2d(12, 64, 2, dilation=2),
          #nn.MaxPool2d(4),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(64, 128),
          nn.ReLU(),
          nn.Linear(128, self.num_outputs),
        )

    def forward(self, x):
        return self.model(x)

    def choose_action(self, state, env):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]

    def train_model(self, online_net, target_net, optimizer, batch):

        states = torch.stack(batch.state).squeeze(1)
        next_states = torch.stack(batch.next_state).squeeze(1)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * self.gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


class Memory(object):
    def __init__(self, capacity):
        self.Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(self.Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = self.Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
