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
    def __init__(self, num_inputs, num_outputs, dev="cpu"):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dev = dev
        self.gamma = 0.97
        '''
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        '''

        self.model = nn.Sequential(
          nn.Conv2d(1, 32, 4, stride=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(32, 128, 2, stride=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Flatten(),
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, self.num_outputs),
        )
        self.model = self.model.to(self.dev)

    def forward(self, x):
        return self.model(x)

    def save(self, gen):
        torch.save(self.model.state_dict(), "models/"+str(gen)+".pt")

    def load(self, gen):
        self.model.load_state_dict(torch.load("models/"+str(gen)+".pt", map_location="cpu"))
        self.model.eval()

    def choose_action(self, state, env):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]

    def train_model(self, online_net, target_net, optimizer, batch):

        states = torch.stack(batch.state).squeeze(1).to(self.dev)
        next_states = torch.stack(batch.next_state).squeeze(1).to(self.dev)
        actions = torch.Tensor(batch.action).float().to(self.dev)
        rewards = torch.Tensor(batch.reward).to(self.dev)
        masks = torch.Tensor(batch.mask).to(self.dev)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        # state q value
        pred = torch.sum(pred.mul(actions), dim=1)

        # new state max possible q value
        target = rewards + masks * self.gamma * next_pred.max(1)[0]

        # Bellmann equation loss
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
