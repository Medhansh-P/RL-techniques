
import numpy as np
from gridworld import GridWorld


def generate_episode(env, gamma=0.9, max_steps=100):

    episode = []

    state = env.reset()

    for _ in range(max_steps):

        action = np.random.randint(env.n_actions)

        next_state, reward, done = env.step(action)

        episode.append((state, reward))

        state = next_state

        if done:
            break

    return episode



def monte_carlo_prediction(
    env,
    episodes=10000,
    gamma=0.9,
    alpha=0.1,
    first_visit=True,
    incremental_mean=False
):

    V = np.zeros(env.n_states)

    returns_sum = np.zeros(env.n_states)
    returns_count = np.zeros(env.n_states)

    for ep in range(episodes):

        episode = generate_episode(env, gamma)

        # Calculate returns backwards
        G = 0
        returns = [0] * len(episode)

        for t in reversed(range(len(episode))):

            state, reward = episode[t]
            G = reward + gamma * G

            returns[t] = G

        visited = set()

        # Apply updates chronologically
        for t, (state, reward) in enumerate(episode):

            index = env.state_to_index(state)

            if first_visit and index in visited:
                continue

            visited.add(index)

            G = returns[t]

            if incremental_mean:

                returns_sum[index] += G
                returns_count[index] += 1

                V[index] = (
                    returns_sum[index] /
                    returns_count[index]
                )

            else:

                V[index] += alpha * (G - V[index])

    return V


if __name__ == "__main__":

    env = GridWorld()

    methods = [
        ("Every Visit - Constant Alpha", False, False),
        ("First Visit - Constant Alpha", True, False),
        ("Every Visit - Incremental Mean", False, True),
        ("First Visit - Incremental Mean", True, True)
    ]

    for name, first, mean in methods:

        values = monte_carlo_prediction(
            env,
            episodes=10000,
            first_visit=first,
            incremental_mean=mean
        )

        print("\n", name)
        print(values.reshape(5, 5))
