# CSE546FinalProject
### Final project for CSE546 Reinforcement Learning: Training a RL agent to rescue people in a forest fire

## Environment Definitions
### Actions
The agent will have four actions available:
- Move Left
- Move Right
- Move Down
- Move Up

### States
The environment will be a gridworld where a cell can contain:
- Nothing
- The Agent
- Fire
- A Person in Need of Rescue

At each time step, the fire will have a chance to spread to neighboring cells. This is done by maintaining a seperate matrix containing the 'heat' values at the corresponding location in the grid environment. Once these values reach a certain threshold, that cell will engulf in flames.

If a cell containing a person sets fire, the person is elimanted from the environment.

The number of people in a grid will be calculated depending on the size of the grid.

### Observations
At each step, the agent will get a vector of length 8. The first four values in the vector, will be intensity values of how close a person is in the corresponding direction. The next four will be the intensity values of how close a fire is in that direction. These intensity values will be dependant on the number of people/fires in the corresponding direction.

### The Agent
The Agent's goal is to pick up all of the people before the fire eliminates them and to escape to safety. When the agent lands on a cell that contains a person, it will automatically pick them up. When the agent no longer has people to pick up (which would be known when the first four values in the observation vector are zero) the agent will need to escape which can be done by moving to the edge of the grid and then moving off of the grid environment.

### Termination
The episode will terminate if the agent traverses onto a cell that contains fire. If the agent makes a move that would result in it leaving the grid world, the episode also terminates because this symbolizes the agent leaving to safety.

### Rewards
The agent is given a small positive reward for picking up any person on the grid world. Since the goal is for the agent to escort the people to safety, an even greater reward is given after the agent escapes the grid world with people (the reward is multiplied by the number of people the agent has picked up). The agent will be given a negative reward if it moves onto a cell that is engulfed in flames.
