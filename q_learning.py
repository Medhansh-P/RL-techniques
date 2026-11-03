
import numpy as np
from gridworld import GridWorld
from sarsa import epsilon_greedy


def train_q_learning(
    env,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    max_steps=100
):

    Q = np.zeros((env.n_states, env.n_actions))

    steps_per_episode = []

    for ep in range(episodes):

        state = env.reset()
        s = env.state_to_index(state)

        steps = 0

        for _ in range(max_steps):

            action = epsilon_greedy(Q, s, epsilon)

            next_state, reward, done = env.step(action)

            ns = env.state_to_index(next_state)

            steps += 1

            if done:

                target = reward

            else:

                target = reward + gamma * np.max(Q[ns])

            td_error = target - Q[s, action]

            Q[s, action] += alpha * td_error

            state = next_state
            s = ns

            if done:
                break

        steps_per_episode.append(steps)

    return Q, steps_per_episode


if __name__ == "__main__":

    env = GridWorld()

    Q, steps = train_q_learning(env)

    print("Q-learning Q-table:")
    print(Q)

    print("Last 10 episode steps:")
    print(steps[-10:])
