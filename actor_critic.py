import torch
from torch import nn, optim
from torch.distributions import Categorical

# Class to define actor network
class Actor(nn.Module):
    def __init__(self, obs_space, action_space, hidden_units=64, dense=False):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.dense = dense
        self.input = nn.Linear(self.obs_space, self.hidden_units)
        self.hidden = nn.Linear(self.hidden_units, self.hidden_units)
        self.output = nn.Linear(self.hidden_units, self.action_space)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs):
        x = self.relu(self.input(obs))
        if self.dense:
            x = self.relu(self.hidden(x))
        x = self.softmax(self.output(x))
        return x

    def choose_action(self, obs):
        probs = self(obs)
        m = Categorical(probs)
        action = m.sample()
        log_probabilities = m.log_prob(action)
        return action[0].item(), log_probabilities

# Class to define critic network
class Critic(nn.Module):
    def __init__(self, obs_space, hidden_units=64, dense=False):
        super().__init__()
        self.obs_space = obs_space
        self.hidden_units = hidden_units
        self.dense = dense
        self.input = nn.Linear(self.obs_space, self.hidden_units)
        self.hidden = nn.Linear(self.hidden_units, self.hidden_units)
        self.output = nn.Linear(self.hidden_units, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, obs):
        x = self.relu(self.input(obs))
        if self.dense:
            x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

# Class for ActorCritic
class ActorCritic:
    def __init__(self, obs_space, action_space, gamma=0.99, actor_lr=1e-2, critic_lr=1e-2, dense=False):
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor = Actor(obs_space, action_space, dense=dense)
        self.critic = Critic(obs_space, dense=dense)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

    def learn(self, obs, action, log_probability, reward, new_obs, done):
        new_critic_value = self.critic(new_obs)
        critic_value = self.critic(obs)
        td_target = reward + self.gamma*new_critic_value.detach()*(0 if done else 1)
        critic_loss = self.loss_fn(critic_value, td_target)
        actor_loss = -log_probability * (td_target - critic_value.detach())
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def choose_action(self, obs):
        return self.actor.choose_action(obs)