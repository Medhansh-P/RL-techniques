
import numpy as np
from gridworld import GridWorld


def epsilon_greedy(Q, state, epsilon):

    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])

    best_actions = np.flatnonzero(
        Q[state] == np.max(Q[state])
    )

    return np.random.choice(best_actions)


def train_sarsa(
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

        action = epsilon_greedy(Q, s, epsilon)

        steps = 0

        for _ in range(max_steps):

            next_state, reward, done = env.step(action)

            ns = env.state_to_index(next_state)

            steps += 1

            if done:

                target = reward

                td_error = target - Q[s, action]

                Q[s, action] += alpha * td_error

                break

            next_action = epsilon_greedy(
                Q, ns, epsilon
            )

            target = reward + gamma * Q[ns, next_action]

            td_error = target - Q[s, action]

            Q[s, action] += alpha * td_error

            state = next_state
            s = ns
            action = next_action

        steps_per_episode.append(steps)

    return Q, steps_per_episode


if __name__ == "__main__":

    env = GridWorld()

    Q, steps = train_sarsa(env)

    print("SARSA Q-table:")
    print(Q)

    print("Last 10 episode steps:")
    print(steps[-10:])
