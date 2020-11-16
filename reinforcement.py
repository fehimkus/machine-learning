
import numpy as np

R = np.matrix([[-1, -1, -1, -1, 0, -1]
              , [-1, -1, -1, 0, -1, 100]
              , [-1, -1, -1, 0, -1, -1]
              , [-1, 0, 0, -1, 0, -1]
              , [-1, 0, 0, -1, -1, 100]
              , [-1, 0, -1, -1, 0, 100]])

Q = np.matrix(np.zeros([6], [6])

gamma = 0.8
initial_state = 1


def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


available_act = available_actions(initial_state)


def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act, 1))
    return next_action


action = sample_next_action(available_act)


def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
        max_value = Q[action, max_index]
        Q[current_state, action] = R[current_state, action] + gamma * max_value


update(initial_state, action, gamma)

for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, gamma)

print(Q / np.max(Q) * 100)
