import math
from typing import Optional

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

class Twenty48determ(gym.Env):
    """
    ### Description
    The 2048 game is a single-player sliding block puzzle game where the goal is to combine tiles to create a tile with
    the number 2048. The game is played on a 4x4 grid with numbered tiles that slide when a player moves them using the
    four arrow keys. Every turn, a new tile will randomly appear in an empty space on the board with a value of either
    two or four. Tiles slide as far as possible in the chosen direction until they are stopped by either another tile or
    the edge of the grid. If two tiles of the same number collide while moving, they will merge into a tile with the
    total value of the two tiles that collided. The resulting tile cannot merge with another tile again in the same
    move. Once a move has been made, a new tile will appear on the board. The game is won when a tile with the number
    2048 appears on the board, hence the name of the game. The game is lost when there are no more empty spaces on the
    board and no legal moves remain. 
    ### Observation Space
    The observation is a `ndarray` with shape `(4,4)` where the elements correspond to the value of the tiles in the
    grid. The value of the tiles are represented by powers of 2. The value of the tiles are as follows:
    | Num | Observation |
    |-----|-------------|
    | 0   | 0           |
    | 1   | 2           |
    | 2   | 4           |
    | 3   | 8           |
    | 4   | 16          |
    | 5   | 32          |
    | 6   | 64          |
    | 7   | 128         |
    | 8   | 256         |
    | 9   | 512         |
    | 10  | 1024        |
    | 11  | 2048        |
    ### Action Space
    The action space is a `Discrete` space with 4 elements. The actions are as follows:
    | Num | Action |
    |-----|--------|
    | 0   | Up     |
    | 1   | Down   |
    | 2   | Right  |
    | 3   | Left   |
    ### Reward:
    The reward is 1 if there is a tile with the value 2048 on the board and 0 otherwise.
    ### Starting State
    The values of two randomly chosen tiles are randomly generated at the start of each episode. The values of the tiles are
    generated using the following procedure:
    1. Generate two sets of two values between 0 and 3 (for tile positions)
    2. If the first set is equal to the second set, generate a new set of two values
    3. Generate two random numbers between 0 and 1 for the values of the tiles
    4. If the number is less than 0.9, the value of the tile is 2. Otherwise, the value of the tile is 4
    ### Transition
    The transition is the result of the action taken by the agent. The transition is as follows:
    1. The agent chooses an action
    2. The tiles slide in the chosen direction until they are stopped by either another tile or the edge of the grid
    3. If two tiles of the same number collide while moving, they will merge into a tile with the total value of the two
    tiles that collided
    4. The resulting tile cannot merge with another tile again in the same move
    5. Once a move has been made, a new tile will be randomly placed on the board with a value of either 2 or 4
    ### Episode End
    The episode ends if either of the following happens:
    1. There is a tile with the value 2048 on the board
    2. There are no more empty spaces on the board and no legal moves remain
    ### Arguments
    ```
    gym.make('Twenty48-v0')
    ```
    ### Version History
    * v0: Initial version release
    * v1: Changed tile spawning from being stochastic to being deterministic
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_tile=2048):
        self.rows = self.cols = 4

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_terminated = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 11, shape=(self.rows*self.cols,), dtype=np.uint8)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        inc_score = 0
        # create a copy of the state
        old_state = np.copy(self.state)
        # moving tiles logic
        if action == RIGHT:
            for row, _ in enumerate(self.state):
                # Move all values to the right
                self._move_right(self.state[row])
                # Merge values right
                inc_score += self._merge_right(self.state[row])
                # Move all values to the right
                self._move_right(self.state[row])
        elif action == LEFT:
            for row, _ in enumerate(self.state):
                # Move all values to the left  
                self._move_left(self.state[row])
                # Merge values left
                inc_score += self._merge_left(self.state[row])
                # Move all values to the left  
                self._move_left(self.state[row])
        elif action == UP:
            # Transpose the board
            self.state = self._transpose_2d_list(self.state)
            for row, _ in enumerate(self.state):
                # Move all values to the left  
                self._move_left(self.state[row])
                # Merge values left
                inc_score += self._merge_left(self.state[row])
                # Move all values to the left  
                self._move_left(self.state[row])
            # Transpose the board back
            self.state = self._transpose_2d_list(self.state)
        elif action == DOWN:
            # Transpose the board
            self.state = self._transpose_2d_list(self.state)
            for row, _ in enumerate(self.state):
                # Move all values to the right
                self._move_right(self.state[row])
                # Merge values right
                inc_score += self._merge_right(self.state[row])
                # Move all values to the right
                self._move_right(self.state[row])
            # Transpose the board back
            self.state = self._transpose_2d_list(self.state)
        else:
            raise Exception("Invalid action")
        
        terminal = self._is_terminal()
        # check if the state has changed
        if np.array_equal(old_state, self.state):
            # state has not changed
            reward = -0.1
        else:
            if not terminal:
                reward = inc_score
            elif self.steps_beyond_terminated is None:
                # Grid is full
                self.steps_beyond_terminated = 0
                reward = inc_score
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward = 0.0

            if not terminal and len(self.get_empty_spaces()) != 0:
                # generate a new tile
                self.spawn_tile()
    
        return np.array(self.state, dtype=np.uint8).flatten(), reward, terminal, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        # initialise state to zeros
        self.state = np.zeros((self.rows,self.cols), dtype=np.uint8)

        # pick the first two positions in self.state
        pos1 = (0, 0)
        pos2 = (0, 1)

        # generate two random numbers between 0 and 1
        rand1 = 0.1
        rand2 = 0.1

        # if the number is less than 0.9, the value of the tile is 2. Otherwise, the value of the tile is 4
        if rand1 < 0.9:
            self.state[pos1[0], pos1[1]] = 1
        else:
            self.state[pos1[0], pos1[1]] = 2
        if rand2 < 0.9:
            self.state[pos2[0], pos2[1]] = 1
        else:
            self.state[pos2[0], pos2[1]] = 2

        self.steps_beyond_terminated = None

        if self.render_mode == "human" or self.render_mode == "ai":
            self.render()
        return np.array(self.state, dtype=np.uint8).flatten(), {}

    def render(self):
        """Render 2D matrix to terminal"""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
            return
        
        if self.render_mode == "human":
            for row in self.state:
                print(row)
            print("-----------------")
        elif self.render_mode == "ai":
            for row in self.state:
                print(row)
            print("-----------------")

        elif self.render_mode == "rgb_array":
            raise NotImplementedError
        
    def get_empty_spaces(self):
        # Get a list of all empty spaces
        empty_spaces = []
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == 0:
                    empty_spaces.append((i, j))
        return empty_spaces
        
    def spawn_tile(self):
        """Spawn a new tile in a random empty space"""
        # Get a list of all empty spaces
        empty_spaces = self.get_empty_spaces()
        # Choose the most top left tile
        new_tile = empty_spaces[0]
        # Spawn a 2 in the empty space
        self.state[new_tile[0]][new_tile[1]] = 1

    def _is_terminal(self):
        if 2048 in np.array(self.state).flatten():
            return True
        else:
            for tile, neighbours in self.matrix_neighbours(self.state):
                if tile == 0:
                    return False
                for neighbour in neighbours:
                    if tile == neighbour:
                        return False
        return True
    
    def _transpose_2d_list(self, matrix):
        return [list(row) for row in zip(*matrix)]

    def _move_left(self, row):
        """Move all values to the left"""
        for i in range(1, 4):
            if row[i]:
                for j in range(i, 0, -1):
                    if not row[j - 1]:
                        row[j - 1] = row[j]
                        row[j] = 0
                    else:
                        break
        return row

    def _move_right(self, row):
        """Move all values to the right"""
        for i in range(2, -1, -1):
            if row[i]:
                for j in range(i, 3):
                    if not row[j + 1]:
                        row[j + 1] = row[j]
                        row[j] = 0
                    else:
                        break
        return row

    def _merge_left(self, row):
        """Merge values from right to left"""
        score = 0
        if row[0] == row[1] and row[0] != 0:
            row[0] += 1
            row[1] = 0
            score += 2 ** row[0] 
        if row[1] == row[2] and row[1] != 0:
            row[1] += 1
            row[2] = 0
            score += 2 ** row[1]
        if row[2] == row[3] and row[2] != 0:
            row[2] += 1
            row[3] = 0
            score += 2 ** row[2]
        return score

    def _merge_right(self, row):
        """Merge values from left to right"""
        score = 0
        if row[2] == row[3] and row[2] != 0:
            row[3] += 1
            row[2] = 0
            score += 2 ** row[3] 
        if row[1] == row[2] and row[1] != 0:
            row[2] += 1
            row[1] = 0
            score += 2 ** row[2] 
        if row[0] == row[1] and row[0] != 0:
            row[1] += 1
            row[0] = 0
            score += 2 ** row[1] 
        return score

    def matrix_neighbours(self, matrix):
        """iterate over elements of 4x4 matrix and get its neighbours"""
        for i in range(4):
            for j in range(4):
                if i == 0:
                    if j == 0:
                        neighbours = [matrix[i][j+1], matrix[i+1][j]]
                    elif j == 3:
                        neighbours = [matrix[i][j-1], matrix[i+1][j]]
                    else:
                        neighbours = [matrix[i][j-1], matrix[i][j+1], matrix[i+1][j]]
                elif i == 3:
                    if j == 0:
                        neighbours = [matrix[i][j+1], matrix[i-1][j]]
                    elif j == 3:
                        neighbours = [matrix[i][j-1], matrix[i-1][j]]
                    else:
                        neighbours = [matrix[i][j-1], matrix[i][j+1], matrix[i-1][j]]
                else:
                    if j == 0:
                        neighbours = [matrix[i][j+1], matrix[i-1][j], matrix[i+1][j]]
                    elif j == 3:
                        neighbours = [matrix[i][j-1], matrix[i-1][j], matrix[i+1][j]]
                    else:
                        neighbours = [matrix[i][j-1], matrix[i][j+1], matrix[i-1][j], matrix[i+1][j]]
                yield matrix[i][j], neighbours

gym.envs.register(
    id='Twenty48-v1',
    entry_point='twenty48v1:Twenty48',
    max_episode_steps=1000,
    reward_threshold=float('inf'),
)