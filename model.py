import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(42)
        "*** YOUR CODE HERE ***"

        fc1_units = 1024 #256
        fc2_units = 1024 #256

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)


    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        action = self.fc3(state)

        return action
