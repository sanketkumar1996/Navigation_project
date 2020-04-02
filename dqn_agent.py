import numpy as np
import random
from collections import namedtuple, deque

from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size (Initially 64)
GAMMA = 0.995           # discount factor (Initially 0.99)
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate (Initially 5e-4)
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size):
        self.epsilon = 1.0
        self.min_eps = 0.05
        self.decay = 0.995
        self.seed = random.seed(42)

        self.state_size = state_size
        self.action_size = action_size

        self.local_nn = Model(state_size, action_size).to(device)
        # self.target_nn = Model(state_size, action_size).to(device)
        self.qnetwork_local.load_state_dict(torch.load('/Users/sanketsans/Documents/Udacity/deepRL/deep-reinforcement-learning/p1_navigation/checkpoint0.pth', map_location=torch.device('cpu')))
        self.qnetwork_local.eval()

        self.optimizer = optim.Adam(self.local_nn.parameters(), lr=LR)

        self.memory = ReplayMemory(BUFFER_SIZE, action_size)

        self.t_step = 0

    def step(self, state, action, reward, new_state, done):
        ## store experiences in replay buffer
        self.memory.add(state, action, reward, new_state, done)

        ##learn every update_every seconds
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if (self.memory.length()) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_nn.eval()
        with torch.no_grad():
            action_values = self.local_nn(state)
        self.local_nn.train()

        self.epsilon = max(self.epsilon*self.decay, self.min_eps)

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        # Get max predicted Q values (for next states) from target model
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.target_nn(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.local_nn(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.local_nn, self.target_nn, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




class ReplayMemory:

    def __init__(self, buffer_size, action_size):
        """Initialize a ReplayBuffer object."""

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(42)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def length(self):
        """Return the current size of internal memory."""
        return len(self.memory)
