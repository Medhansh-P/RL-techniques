
import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld
from sarsa import train_sarsa
from q_learning import train_q_learning


NUM_RUNS = 50
NUM_EPISODES = 500

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1


def run_experiments():

    # Store steps for every run and episode
    sarsa_results = np.zeros((NUM_RUNS, NUM_EPISODES))
    q_results = np.zeros((NUM_RUNS, NUM_EPISODES))

    # Accumulate Q-tables across runs
    sarsa_Q_sum = np.zeros((25, 4))
    q_Q_sum = np.zeros((25, 4))

    for run in range(NUM_RUNS):

        print(f"Running experiment {run + 1}/{NUM_RUNS}")

        # -------------------------
        # SARSA
        # -------------------------

        np.random.seed(run)

        env = GridWorld()

        Q_sarsa, sarsa_steps = train_sarsa(
            env,
            episodes=NUM_EPISODES,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON
        )

        sarsa_results[run] = sarsa_steps

        sarsa_Q_sum += Q_sarsa

        # -------------------------
        # Q-LEARNING
        # -------------------------

        np.random.seed(run)

        env = GridWorld()

        Q_q, q_steps = train_q_learning(
            env,
            episodes=NUM_EPISODES,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON
        )

        q_results[run] = q_steps

        q_Q_sum += Q_q

    # -------------------------
    # AVERAGE RESULTS
    # -------------------------

    avg_sarsa = np.mean(sarsa_results, axis=0)

    avg_q = np.mean(q_results, axis=0)

    avg_sarsa_Q = sarsa_Q_sum / NUM_RUNS

    avg_q_Q = q_Q_sum / NUM_RUNS

    # -------------------------
    # LEARNING CURVE
    # -------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, NUM_EPISODES + 1),
        avg_sarsa,
        label="SARSA",
        linewidth=2
    )

    plt.plot(
        range(1, NUM_EPISODES + 1),
        avg_q,
        label="Q-learning",
        linewidth=2
    )

    plt.xlabel("Episodes")
    plt.ylabel("Average Steps per Episode")

    plt.title(
        "SARSA vs Q-learning (Average of 50 Runs)"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "sarsa_q_learning.png",
        dpi=300
    )

    plt.show()

    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print("\nAverage Steps in First 10 Episodes")

    print("SARSA:", np.round(avg_sarsa[:10], 2))

    print("Q-learning:", np.round(avg_q[:10], 2))

    print("\nAverage Steps in Last 10 Episodes")

    print("SARSA:", np.round(avg_sarsa[-10:], 2))

    print("Q-learning:", np.round(avg_q[-10:], 2))

    print("\nAverage SARSA Q-table:")

    print(np.round(avg_sarsa_Q, 2))

    print("\nAverage Q-learning Q-table:")

    print(np.round(avg_q_Q, 2))

    # -------------------------
    # RETURN RESULTS
    # -------------------------

    return (
        avg_sarsa_Q,
        avg_q_Q,
        avg_sarsa,
        avg_q
    )


if __name__ == "__main__":

    sarsa_Q, q_Q, sarsa_curve, q_curve = run_experiments()
