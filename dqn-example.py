from qagent import QNet, Memory
import gym
import torch
import numpy as np
import torch.optim as optim


env = gym.make("CartPole-v0")  # Create the environment
env.seed(42)

gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
print('state size:', num_inputs)
print('action size:', num_actions)

online_net = QNet(num_inputs, num_actions)
target_net = QNet(num_inputs, num_actions)

online_net.train()
target_net.train()

memory = Memory(capacity=1000)

running_score = 0
epsilon = 1.0
steps = 0
loss = 0

optimizer = optim.Adam(online_net.parameters(), lr=lr)

steps = 0

def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # Target <- Net
    # Copies over the weights
    target_net.load_state_dict(online_net.state_dict())


for e in range(3000):
    done = False

    score = 0
    state = env.reset()
    state = torch.Tensor(state)
    state = state.unsqueeze(0)

    while not done:

        steps += 1

        action = get_action(state, target_net, epsilon, env)

        # need snake equivalent
        next_state, reward, done, _ = env.step(action)  
        if e % 100 == 0:
            env.render()

        #env.render()
        next_state = torch.Tensor(next_state)
        next_state = next_state.unsqueeze(0) 

        mask = 0 if done else 1
        reward = reward if not done or score == 499 else -1
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.push(state, next_state, action_one_hot, reward, mask)

        score += reward
        state = next_state

        if steps > initial_exploration:
            # hyperparameter to balance risk/reward
            epsilon -= 0.00005
            epsilon = max(epsilon, 0.1)

            batch = memory.sample(batch_size)

            loss = online_net.train_model(online_net=online_net, target_net=target_net, 
                                    optimizer=optimizer, batch=batch)

            if steps % update_target == 0:
                update_target_model(online_net, target_net)

    score = score if score == 500.0 else score + 1
    running_score = 0.99 * running_score + 0.01 * score
    print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
        e, running_score, epsilon))

    if running_score > goal_score:
        break