import numpy as np


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


qagent = QAgent(5, 2)

