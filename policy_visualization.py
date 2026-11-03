
import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld


ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}


def show_policy(Q, env, title):

    policy = np.full(
        (env.rows, env.cols),
        "",
        dtype=object
    )

    for r in range(env.rows):
        for c in range(env.cols):

            state = (r, c)

            if state in env.obstacles:
                policy[r, c] = "X"

            elif state == env.goal:
                policy[r, c] = "G"

            elif state == env.start:
                policy[r, c] = "S"

            else:

                s = env.state_to_index(state)

                best_action = np.argmax(Q[s])

                policy[r, c] = ARROWS[best_action]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)

    ax.set_xticks(np.arange(env.cols + 1))
    ax.set_yticks(np.arange(env.rows + 1))

    ax.grid(True)

    ax.invert_yaxis()

    for r in range(env.rows):
        for c in range(env.cols):

            ax.text(
                c + 0.5,
                r + 0.5,
                policy[r, c],
                ha="center",
                va="center",
                fontsize=20
            )

    ax.set_title(title)

    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"

    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":

    env = GridWorld()

    # Import trained Q-tables from experiment runner
    from experiments import run_experiments

    sarsa_Q, q_Q,_,_ = run_experiments()

    show_policy(
        sarsa_Q,
        env,
        "SARSA Greedy Policy"
    )

    show_policy(
        q_Q,
        env,
        "Q Learning Greedy Policy"
    )
