import gym
import gym.spaces as spaces
import numpy as np
import random
import cv2


class ForestFire(gym.Env):
    def __init__(self, height, width):
        # Initializes the class
        # Define action and observation space

        # Setting the grid size
        self.env_height = height
        self.env_width = width
        self.observation_space = spaces.Discrete(self.env_height * self.env_width)

        # Actions: left, right, up, down
        self.action_space = spaces.Discrete(4)

        # Creating the heat-matrix for the fire spreading
        self.heat_matrix = np.zeros(shape=(height, width))

        # Set location values
        self.agent_location = (0, 0)
        self.people_locations = []

        # Compute number of people to put around the map
        self.n_people = int(np.floor(np.sqrt((self.env_height * self.env_width))))

        # Max number of time steps in a run
        # self.max_timesteps = 0

    def step(self, action):
        pass

    def reset(self):
        # Setting agent location
        self.agent_location = (0, 0)

        # Reset heat map
        self.heat_matrix = np.zeros((self.env_height, self.env_width))

        # Set Fire Locations
        self.heat_matrix[int(self.env_height / 2)][0] = 1
        self.heat_matrix[int(self.env_height - 1), int(self.env_width / 2)] = 1

        # Place people around map
        self.people_locations = []
        for i in range(self.n_people):
            x = random.randint(0, self.env_width-1)
            y = random.randint(0, self.env_height-1)
            # Check to make sure person is not already positioned on a cell with something else there
            while (x, y) in self.people_locations or self.heat_matrix[x][y] == 1 or (x, y) == self.agent_location:
                x = random.randint(0, self.env_width-1)
                y = random.randint(0, self.env_height-1)
            self.people_locations.append((x, y))

    def render(self):
        # Create grid to display world on
        gridworld = np.zeros(shape=(self.env_height, self.env_width, 3))

        # Place agent on the grid
        gridworld[self.agent_location] = (255, 0, 0)

        # Place fires on gridworld
        for i in range(len(self.heat_matrix)):
            for j in range(len(self.heat_matrix[0])):
                if self.heat_matrix[i][j] == 1:
                    gridworld[i, j] = (0, 0, 255)

        # Place people on the gridworld
        for person in self.people_locations:
            gridworld[person] = (0, 255, 0)

        cv2.imshow('matrix', gridworld)
        cv2.waitKey(0)


if __name__ == '__main__':
    env = ForestFire(150, 150)
    env.reset()
    env.render()
