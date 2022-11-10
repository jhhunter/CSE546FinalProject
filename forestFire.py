import gym
import gym.spaces as spaces
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
        if(self.agent_location[0] >= self.env_height):
            self.agent_location = (self.env_height-1, self.agent_location[1])
            reward -= 10
        elif(self.agent_location[0] < 0):
            self.agent_location = (0, self.agent_location[1])
            reward -= 10
        if(self.agent_location[1] >= self.env_width):
            self.agent_location = (self.agent_location[0], self.env_width-1)
            reward -= 10
        elif(self.agent_location[1] < 0):
            self.agent_location = (self.agent_location[0], 0)
            reward -= 10

        # Check if the agent moved into fire
        if self.heat_matrix[self.agent_location] == 1:
            # print("Agent in fire")
            done = True
            reward = -10

        # Check if agent picked up a person
        for i in range(self.n_people):
            if self.agent_location == self.people_locations[i]:
                # Agent found person
                # print('Person Found!')
                self.people_found += 1
                self.people_locations.pop(i)
                self.n_people -= 1
                reward += 10
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
                        # print("Person is gone")
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

        # Done if no person left to save
        if(self.n_people <= 0):
            done = True

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

        # Compute number of people to put around the map
        self.n_people = int(np.floor(np.sqrt((self.env_height * self.env_width))))

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

    # If render is not working and showing a QT error, type this in the console: 'export QT_QPA_PLATFORM=offscreen' 

    def render(self, plot=True):
        # Create grid to display world on
        # gridworld = np.zeros(shape=(self.env_height, self.env_width, 3))

        # # Place agent on the grid
        # gridworld[self.agent_location] = (255, 0, 0)

        # # Place fires on gridworld
        # for i in range(len(self.heat_matrix)):
        #     for j in range(len(self.heat_matrix[0])):
        #         if self.heat_matrix[i][j] == 1:
        #             gridworld[i, j] = (0, 0, 255)

        # # Place people on the gridworld
        # for person in self.people_locations:
        #     gridworld[person] = (0, 255, 0)
        
        # gridworld = cv2.resize(gridworld, (500, 500))
        # cv2.imshow('matrix', gridworld)
        # cv2.waitKey(0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_height)
        zoom_val = 0.2
        if(self.heat_matrix[self.agent_location] != 1):
            agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent.png'), zoom=0.05),
                                    np.add(self.agent_location, [0.5, 0.5]), frameon=False)
            ax.add_artist(agent)
        for per in self.people_locations:
            if(self.heat_matrix[per] != 1):
                person = AnnotationBbox(OffsetImage(plt.imread('./images/person.png'), zoom=zoom_val),
                                    np.add(per, [0.5, 0.5]), frameon=False)
                ax.add_artist(person)
        for row in range(len(self.heat_matrix)):
            for col in range(len(self.heat_matrix[0])):
                if(self.heat_matrix[row, col] == 1):
                    fire = AnnotationBbox(OffsetImage(plt.imread('./images/fire.png'), zoom=0.03),
                                    np.add((row, col), [0.5, 0.5]), frameon=False)
                    ax.add_artist(fire)
                else:
                    if((row, col) == self.agent_location or (row, col) in self.people_locations):
                        continue
                    tree = AnnotationBbox(OffsetImage(plt.imread('./images/tree.png'), zoom=zoom_val),
                                    np.add((row, col), [0.5, 0.5]), frameon=False)
                    ax.add_artist(tree)
        
        plt.xticks(range(self.env_width))
        plt.yticks(range(self.env_height))
        plt.grid()  # Setting the plot to be of the type 'grid'.

        if plot:  # Displaying the plot.
            plt.show()
        else:  # Returning the preprocessed image representation of the environment.
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())#[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = 512
            height = 512
            dim = (width, height)
            # noinspection PyUnresolvedReferences
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            plt.show()
            return preprocessed_image


if __name__ == '__main__':
    env = ForestFire(10, 10)
    obs = env.reset()
    env.render()
    done = False
    while not done:
        obs, rew, done = env.step(2)
        env.render()
