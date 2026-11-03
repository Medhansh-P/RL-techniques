
import numpy as np
from gridworld import GridWorld


def td_prediction(
    env,
    episodes=10000,
    alpha=0.1,
    gamma=0.9,
    max_steps=100
):

    V = np.zeros(env.n_states)

    for episode in range(episodes):

        state = env.reset()

        for _ in range(max_steps):

            action = np.random.randint(env.n_actions)

            next_state, reward, done = env.step(action)

            s = env.state_to_index(state)
            ns = env.state_to_index(next_state)

            if done:
                target = reward
            else:
                target = reward + gamma * V[ns]

            td_error = target - V[s]

            V[s] += alpha * td_error

            state = next_state

            if done:
                break

    return V


if __name__ == "__main__":

    env = GridWorld()

    V = td_prediction(env)

    print("TD(0) Value Function:")
    print(V.reshape(5, 5))
