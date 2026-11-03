# Reinforcement Learning Assignment 2

## Monte Carlo, Temporal Difference Learning and Value Function Approximation

**Course:** Reinforcement Learning  
**Assignment:** Assignment 2  
**Language:** Python 3  
**Libraries:** NumPy, Matplotlib

---

## 1. Introduction

This project implements fundamental Reinforcement Learning algorithms using a custom GridWorld environment.

The objective is to understand how an agent learns to predict state values, optimize its actions and approximate value functions through experience.

The assignment is divided into three parts:

1. Monte Carlo Prediction and TD(0) Prediction.
2. SARSA and Q-learning for Reinforcement Learning control.
3. Monte Carlo and TD(0) with Value Function Approximation.

All algorithms and the GridWorld environment are implemented from scratch without using Gym, Gymnasium or any prebuilt Reinforcement Learning libraries.

---

## 2. Project Objectives

- Implement a custom GridWorld environment.
- Understand first-visit and every-visit Monte Carlo prediction.
- Compare constant learning rate and incremental mean updates.
- Implement TD(0) prediction.
- Implement SARSA and Q-learning using epsilon-greedy exploration.
- Compare learning curves averaged over 50 independent runs.
- Visualize the greedy policies learned by both control algorithms.
- Implement value function approximation using shared feature weights.
- Compare Monte Carlo and TD(0) approximation using MAE.

---

## 3. GridWorld Environment

A custom 5 × 5 GridWorld environment is used.

### Environment Configuration

| Parameter         | Value                      |
| ----------------- | -------------------------- |
| Grid size         | 5 × 5                      |
| Number of states  | 25                         |
| Number of actions | 4                          |
| Starting state    | (0, 0)                     |
| Goal state        | (4, 4)                     |
| Obstacles         | (1,1), (1,3), (3,0), (3,2) |
| Discount factor   | 0.9                        |
| Learning rate     | 0.1                        |
| Exploration rate  | 0.1                        |

### Available Actions

| Action | Direction |
| ------ | --------- |
| 0      | Up        |
| 1      | Down      |
| 2      | Left      |
| 3      | Right     |

### Reward Structure

| Event              | Reward |
| ------------------ | -----: |
| Normal movement    |     -1 |
| Reaching goal      |    +10 |
| Obstacle collision |    -10 |
| Boundary collision |     -1 |

Obstacles are treated as blocked cells. When the agent attempts to enter an obstacle, it remains in its current state.

The goal is a terminal state. Episodes are also subject to a maximum step limit to prevent infinite loops.

---

## 4. Algorithms Implemented

### Part 1: Monte Carlo and TD Prediction

#### A. Monte Carlo Prediction

Monte Carlo prediction estimates the state-value function using complete episodes.

The return is calculated as:

G(t) = R(t+1) + γR(t+2) + γ²R(t+3) + ...

The following variants are implemented:

- First-visit MC with constant alpha.
- Every-visit MC with constant alpha.
- First-visit MC with incremental mean.
- Every-visit MC with incremental mean.

Constant alpha update:

V(s) = V(s) + α[G - V(s)]

Incremental mean update:

V(s) = V(s) + (1/N(s))[G - V(s)]

#### B. TD(0) Prediction

TD(0) updates state values after every transition using one-step bootstrapping.

Update equation:

V(s) = V(s) + α[R + γV(s') - V(s)]

Unlike Monte Carlo, TD(0) does not wait for an episode to finish.

---

### Part 2: SARSA and Q-learning

Both algorithms learn action-value functions using epsilon-greedy exploration.

Parameters:

- Epsilon = 0.1
- Alpha = 0.1
- Gamma = 0.9
- Independent runs = 50
- Episodes per run = 500

#### A. SARSA

SARSA is an on-policy TD control algorithm.

It uses the actual next action selected by the behavior policy.

Update equation:

Q(s,a) = Q(s,a) + α[R + γQ(s',a') - Q(s,a)]

#### B. Q-learning

Q-learning is an off-policy TD control algorithm.

It uses the maximum estimated Q-value of the next state.

Update equation:

Q(s,a) = Q(s,a) + α[R + γ max Q(s',a) - Q(s,a)]

#### Comparison

The learning curves are averaged over 50 independent runs.

The project also visualizes the greedy policies learned by SARSA and Q-learning.

---

### Part 3: Value Function Approximation

Value Function Approximation is used to estimate state values using shared parameters instead of storing an independent value for every state.

Linear approximation:

V(s,w) = wᵀx(s)

Where:

- w represents the weight vector.
- x(s) represents the feature vector.
- V(s,w) represents the estimated state value.

The planned feature representation uses:

1. Bias feature.
2. Normalized row position.
3. Normalized column position.
4. Normalized Manhattan distance to the goal.

#### A. Monte Carlo with VFA

Weight update:

w = w + α[G - V(s,w)]x(s)

#### B. TD(0) with VFA

Weight update:

w = w + αδx(s)

Where:

δ = R + γV(s',w) - V(s,w)

#### Experimental Configuration

- 100 episodes.
- Identical initial weights for both algorithms.
- Identical fixed state trajectory and rewards.
- Learned weights are retained between episodes.
- MC and TD estimates are compared after episodes 1, 10, 25, 50 and 100.
- MAE is plotted against episode number.

The same reference values must be used when calculating MAE for both algorithms.

---

## 5. Project Structure

```text
RL_Assignment_2/
│
├── gridworld.py
│
├── monte_carlo.py
│
├── td_prediction.py
│
├── sarsa.py
│
├── q_learning.py
│
├── experiments.py
│
├── policy_visualization.py
│
├── value_function_approximation.py
│
├── results/
│   ├── mc_prediction.png
│   ├── td_prediction.png
│   ├── sarsa_q_learning.png
│   ├── sarsa_greedy_policy.png
│   ├── q_learning_greedy_policy.png
│   ├── vfa_estimates.png
│   └── vfa_mae.png
│
├── README.md
│
└── requirements.txt
```

---

## 6. Installation and Requirements

### Prerequisites

- Python 3.9 or later.
- NumPy.
- Matplotlib.

### Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy matplotlib
```

---

## 7. How to Run

Run the individual modules from the project directory.

### Monte Carlo Prediction

```bash
python monte_carlo.py
```

### TD(0) Prediction

```bash
python td_prediction.py
```

### SARSA and Q-learning Experiments

```bash
python experiments.py
```

This runs 50 independent experiments and generates the average learning curve.

### Greedy Policy Visualization

```bash
python policy_visualization.py
```

### Value Function Approximation

```bash
python value_function_approximation.py
```

Note: Run the VFA experiment after adding its implementation and corresponding reference-value evaluation.

---

## 8. Expected Outputs

The project generates the following experimental results:

| Output                    | Description                              |
| ------------------------- | ---------------------------------------- |
| MC value estimates        | State values from four MC variants       |
| TD value estimates        | State values estimated using TD(0)       |
| SARSA learning curve      | Average steps per episode                |
| Q-learning learning curve | Average steps per episode                |
| Greedy SARSA policy       | Learned action for each grid state       |
| Greedy Q-learning policy  | Learned action for each grid state       |
| VFA comparison table      | MC and TD estimates at selected episodes |
| VFA MAE plot              | Prediction error over 100 episodes       |

---

## 9. Experimental Observations

The algorithms use different update mechanisms:

- Monte Carlo learns from complete episode returns.
- TD(0) uses immediate rewards and estimated next-state values.
- SARSA updates using the next action actually selected by its behavior policy.
- Q-learning updates using the maximum next-state action value.
- Value Function Approximation allows multiple states to share parameters.

MC and TD may produce different learning trajectories even when they use the same initial weights and experience because their update targets are different.

The numerical observations and generated plots should be added after running the experiments.

---

## 10. Assumptions and Limitations

1. The GridWorld environment is deterministic.
2. Obstacles are blocked cells rather than terminal states.
3. Reaching the goal terminates an episode.
4. A maximum step limit prevents episodes from running indefinitely.
5. Random policies are used for the prediction experiments.
6. Epsilon-greedy exploration is used for SARSA and Q-learning.
7. Independent runs use reproducible random seeds.
8. Value Function Approximation uses linear features.
9. No external Reinforcement Learning framework is used.

---

## 11. Technologies Used

- Python
- NumPy
- Matplotlib
- Git and GitHub

No Gym, Gymnasium, Stable-Baselines3 or other prebuilt Reinforcement Learning libraries are used.

---

## 12. Conclusion

This assignment demonstrates important Reinforcement Learning techniques, including Monte Carlo prediction, Temporal Difference learning, SARSA, Q-learning and Value Function Approximation.

It highlights the differences between episodic and incremental learning, on-policy and off-policy control, and tabular versus approximate value functions.

The experiments help understand how an agent improves its decisions through interaction and experience.

---

**End of Assignment 2**
