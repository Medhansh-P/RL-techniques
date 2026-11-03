
import numpy as np

class GridWorld:

    def __init__(self):
        self.rows = 5
        self.cols = 5

        self.start = (0, 0)
        self.goal = (4, 4)

        self.obstacles = {
            (1, 1), (1, 3),
            (3, 0), (3, 2)
        }

        # 0 = Up, 1 = Down, 2 = Left, 3 = Right
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        self.n_actions = 4
        self.n_states = self.rows * self.cols
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):

        dr, dc = self.actions[action]

        r, c = self.state

        nr = r + dr
        nc = c + dc

        # Check grid boundaries
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            reward = -1
            next_state = self.state

        else:
            next_state = (nr, nc)

            if next_state in self.obstacles:
                reward = -10
                next_state = self.state

            elif next_state == self.goal:
                reward = 10

            else:
                reward = -1

        self.state = next_state

        done = (self.state == self.goal)

        return self.state, reward, done

    def state_to_index(self, state):
        return state[0] * self.cols + state[1]

    def index_to_state(self, index):
        return (index // self.cols, index % self.cols)

    def display(self):

        grid = np.zeros((self.rows, self.cols), dtype=int)

        for r, c in self.obstacles:
            grid[r, c] = -1

        grid[self.start] = 2
        grid[self.goal] = 3

        print(grid)
