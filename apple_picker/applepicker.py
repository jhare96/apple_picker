import gym
import numpy as np
import random
from pathlib import Path
import copy


class ApplePicker(gym.Env):
    def __init__(self, map_filepath='orchid.npy', num_objects=100, default_reward=-0.1):
        super(ApplePicker, self)
        self.action_space = gym.spaces.Discrete(4)
        self._grid = (np.load(Path(__file__).parent / f'maps/{map_filepath}') * 255).astype(np.uint8)
        self.state = self._grid.copy()
        self.item_locs = {}
        self.nobjects = num_objects
        self.agent_loc = self.rand_loc()
        self.reset()
        self.def_reward = default_reward
        self.max_row = self._grid.shape[0]-1
        self.max_col = self._grid.shape[1]-1
        print(self._grid.dtype)

    def rand_loc(self):
        while True:
            x = np.random.randint(0, high=self._grid.shape[0])
            y = np.random.randint(0, high=self._grid.shape[1])
            if self._grid[x,y,2] == 0:
                return x,y

    def generate_random_locs(self, num_objects):
        objects = 0
        while objects < num_objects:
            x, y = self.rand_loc()
            if x in self.item_locs:
                if y in self.item_locs:
                    pass
                else:
                    self.item_locs[x][y] = 1
                    objects += 1
                    self.state[x,y,1] = 255
            else:
                self.item_locs[x] = {y:1}
                objects += 1
                self.state[x,y,1] = 255
    
    def step(self, action):
        if action > 3:
            raise ValueError('action size if 4')
        row, column = self.agent_loc
        if action == 0: # up
            next_agent_loc = [row-1, column]
        if action == 1: # down
            next_agent_loc = [row+1, column]
        if action == 2: # left
            next_agent_loc = [row, column-1]
        if action == 3: # right
            next_agent_loc = [row, column+1]

        reward = self.def_reward
        next_row, next_column = next_agent_loc
        if (0 <= next_row <= self.max_row) and (0 <= next_column <= self.max_col) and self.state[next_row, next_column, 2] == 0:
            try:
                del self.item_locs[next_row][next_column]
                self.item_locs = {k:v for k,v in self.item_locs.items() if len(v)>0}
                reward = 1
            except KeyError:
                pass

           # self.agent_loc = next_agent_loc
            self.update_state(next_agent_loc)

        info = {}
        done = True if len(self.item_locs) == 0 else False
        return self.state, reward, done, info
        
    def update_state(self, next_agent_loc):
        row, column = self.agent_loc
        next_row, next_column = next_agent_loc
        self.state[row, column] = self._grid[row, column]
        self.state[next_row, next_column, 0] = 255
        self.agent_loc = next_agent_loc
    
    def reset(self):
        self.state = self._grid.copy()
        self.item_locs = {}
        agent_loc = self.rand_loc()
        self.generate_random_locs(self.nobjects)
        self.update_state(agent_loc)
        return self.state



class ApplePickerDeterministic(ApplePicker):
    def __init__(self, num_objects=20, default_reward=0, map_filepath='orchid.npy'):
        super(ApplePicker, self)
        self.action_space = gym.spaces.Discrete(4)
        self._grid = (np.load(Path(__file__).parent / f'maps/{map_filepath}') * 255).astype(np.uint8)
        self.max_row = self._grid.shape[0]-1
        self.max_col = self._grid.shape[1]-1
        self.state = self._grid.copy()
        self.item_locs = {}
        self.nobjects = num_objects
        self.start_loc = self.rand_loc()
        self.agent_loc = self.start_loc
        self.generate_random_locs(self.nobjects)
        self.def_reward = default_reward
        self.item_locs_master = copy.deepcopy(self.item_locs)
        self.reset()

    def _draw_apples(self):
        for row, columns in self.item_locs.items():
            for col in columns:
                self.state[row, col, 1] = 255

    def reset(self):
        self.item_locs = copy.deepcopy(self.item_locs_master)
        self.state = self._grid.copy()
        self._draw_apples()
        self.agent_loc = self.start_loc
        self.update_state(self.start_loc)
        return self.state


       

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = ApplePicker(num_objects=100)
    obs = env.reset()
    plt.imshow(obs)
    for i in range(1000):
        obs, reward, done, info = env.step(random.choice([0,1,2,3]))
        if reward > 0:
            print('reward', reward)
        if done:
            print('done')
            break
            
    plt.figure()
    plt.imshow(obs)
    

    obs = env.reset()
    plt.figure()
    plt.imshow(obs)
    plt.show()
