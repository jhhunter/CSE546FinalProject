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

        low = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            dtype=np.float32
        )

        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Actions: left, right, up, down
        self.action_space = spaces.Discrete(4)

        # Creating the heat-matrix for the fire spreading
        self.heat_matrix = np.zeros(shape=(height, width))

        # Set location values
        self.agent_location = (0, 0)
        self.people_locations = []

        # Compute number of people to put around the map
        self.n_people = int(np.floor(np.sqrt((self.env_height * self.env_width))))

        # Value to keep track of people found
        self.people_found = 0

        # Max number of time steps in a run
        # self.max_timesteps = 0

    def step(self, action):
        # Move the agent's location
        if action == 0:
            # Move Up
            self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
        elif action == 1:
            # Move Right
            self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        elif action == 2:
            # Move Down
            self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])
        elif action == 3:
            # Move Left
            self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
        else:
            print("ERROR: Invalid Action")

        done = False
        reward = 0
        observation = np.zeros(8)
        # Check if agent left the map
        if self.agent_location[0] >= self.env_height or env.agent_location[0] < 0 or self.agent_location[1] >= self.env_width or env.agent_location[1] < 0:
            print("Agent Left")
            for i in range(self.people_found):
                reward += 100
            done = True
            return observation, reward, done

        # Check if the agent moved into fire
        if self.heat_matrix[self.agent_location] == 1:
            print("Agent in fire")
            done = True
            reward = -10

        # Check if agent picked up a person
        for i in range(self.n_people):
            if self.agent_location == self.people_locations[i]:
                # Agent found person
                print('Person Found!')
                self.people_found += 1
                self.people_locations.pop(i)
                self.n_people -= 1
                break

        # Increment the fire map
        for i in range(len(self.heat_matrix)):
            for j in range(len(self.heat_matrix[0])):
                if self.heat_matrix[i][j] == 1:
                    # Increment heat at each adjacent location
                    if i - 1 >= 0 and self.heat_matrix[i - 1][j] < 1:
                        self.heat_matrix[i - 1][j] += random.random()
                    if i + 1 < self.env_height and self.heat_matrix[i + 1][j] < 1:
                        self.heat_matrix[i + 1][j] += random.random()
                    if j - 1 >= 0 and self.heat_matrix[i][j - 1] < 1:
                        self.heat_matrix[i][j - 1] += random.random()
                    if j + 1 < self.env_width and self.heat_matrix[i][j + 1] < 1:
                        self.heat_matrix[i][j + 1] += random.random()

        for i in range(len(self.heat_matrix)):
            for j in range(len(self.heat_matrix[0])):
                if self.heat_matrix[i][j] > 1:
                    self.heat_matrix[i][j] = 1

        # Determine if a person has been lost in the fire
        for person in self.people_locations:
            if self.heat_matrix[person] == 1:
                # Person is in fire and must be removed
                for per in range(len(self.people_locations)):
                    if self.people_locations[per] == person:
                        self.people_locations.pop(per)
                        self.n_people -= 1
                        print("Person is gone")
                        break

        # Calculate observation
        up = 0
        right = 1
        down = 2
        left = 3
        agent_row, agent_col = self.agent_location
        for per_row, per_col in self.people_locations:
            row_off = agent_row - per_row
            col_off = agent_col - per_col
            if(row_off > 0):
                # Agent row is greater than person row. Contribution guaranteed in up.
                observation[up] += 1/row_off
            elif(row_off < 0):
                # Agent row is smaller than person row. Contribution guaranteed in down.
                observation[down] += 1/abs(row_off)
            if(col_off > 0):
                # Agent column is greater than person column. Contribution guaranteed in left.
                observation[left] += 1/col_off
            elif(col_off < 0):
                # Agent column is smaller than person column. Contribution guaranteed in right.
                observation[right] += 1/abs(col_off)

        for row in range(len(self.heat_matrix)):
            for col in range(len(self.heat_matrix[0])):
                if self.heat_matrix[row][col] == 1:
                    row_off = agent_row - row
                    col_off = agent_col - col
                    if(row_off > 0):
                        # Agent row is greater than fire row. Contribution guaranteed in up.
                        observation[4+up] += 1/row_off
                    elif(row_off< 0):
                        # Agent row is smaller than fire row. Contribution guaranteed in down.
                        observation[4+down] += 1/abs(row_off)
                    if(col_off > 0):
                        # Agent column is greater than fire column. Contribution guaranteed in left.
                        observation[4+left] += 1/col_off
                    elif(col_off < 0):
                        # Agent column is smaller than fire column. Contribution guaranteed in right.
                        observation[4+right] += 1/abs(col_off)

        self.state = observation 
        return self.state, reward, done

    def reset(self):
        # Setting agent location
        self.agent_location = (0, 0)

        # Reset heat map
        self.heat_matrix = np.zeros((self.env_height, self.env_width))

        # Set Fire Locations
        self.heat_matrix[int(self.env_height / 2)][0] = 1
        self.heat_matrix[int(self.env_height - 1), int(self.env_width / 2)] = 1

        # Set amount of people found
        self.people_found = 0

        # Place people around map
        self.people_locations = []
        for i in range(self.n_people):
            col = random.randint(0, self.env_width-1)
            row = random.randint(0, self.env_height-1)
            # Check to make sure person is not already positioned on a cell with something else there
            while (row, col) in self.people_locations or self.heat_matrix[row][col] == 1 or (row, col) == self.agent_location:
                col = random.randint(0, self.env_width-1)
                row = random.randint(0, self.env_height-1)
            self.people_locations.append((row, col))
        
        # Calculating observation
        observation = np.zeros(8)
        up = 0
        right = 1
        down = 2
        left = 3
        agent_row, agent_col = self.agent_location
        for per_row, per_col in self.people_locations:
            row_off = agent_row - per_row
            col_off = agent_col - per_col
            if(row_off > 0):
                # Agent row is greater than person row. Contribution guaranteed in up.
                observation[up] += 1/row_off
            elif(row_off< 0):
                # Agent row is smaller than person row. Contribution guaranteed in down.
                observation[down] += 1/abs(row_off)
            if(col_off > 0):
                # Agent column is greater than person column. Contribution guaranteed in left.
                observation[left] += 1/col_off
            elif(col_off < 0):
                # Agent column is smaller than person column. Contribution guaranteed in right.
                observation[right] += 1/abs(col_off)

        
        for row in range(len(self.heat_matrix)):
            for col in range(len(self.heat_matrix[0])):
                if self.heat_matrix[row][col] == 1:
                    row_off = agent_row - row
                    col_off = agent_col - col
                    if(row_off > 0):
                        # Agent row is greater than fire row. Contribution guaranteed in up.
                        observation[4+up] += 1/row_off
                    elif(row_off< 0):
                        # Agent row is smaller than fire row. Contribution guaranteed in down.
                        observation[4+down] += 1/abs(row_off)
                    if(col_off > 0):
                        # Agent column is greater than fire column. Contribution guaranteed in left.
                        observation[4+left] += 1/col_off
                    elif(col_off < 0):
                        # Agent column is smaller than fire column. Contribution guaranteed in right.
                        observation[4+right] += 1/abs(col_off)

        self.state = observation
        return self.state

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
        
        gridworld = cv2.resize(gridworld, (500, 500))
        cv2.imshow('matrix', gridworld)
        cv2.waitKey(0)


if __name__ == '__main__':
    env = ForestFire(150, 150)
    obs = env.reset()
    env.render()
    done = False
    while not done:
        obs, rew, done = env.step(2)
        env.render()
