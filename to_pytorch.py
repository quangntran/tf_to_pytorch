import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.models import load_model
import time
import numpy as np
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(3, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 3)
        # self.fc5 = nn.Linear(h, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        return self.fc4(x)
X, Y, Z = 7,8,90
net=Actor(time.time())

model = load_model('model1.h5')
pred = np.array([[X, Y, Z]])
print(model.predict(pred))
# model = load_model('simpleModel.h5')
weights=model.get_weights()

import pickle
with open('model1.pkl', 'wb') as f:
    pickle.dump(weights, f)

weights = pickle.load( open( "model1.pkl", "rb" ) )

"""
Step 3: Assign those weights to your pytorch model
"""

net.fc1.weight.data=torch.from_numpy(np.transpose(weights[0]))
net.fc1.bias.data=torch.from_numpy(weights[1])
net.fc2.weight.data=torch.from_numpy(np.transpose(weights[2]))
net.fc2.bias.data=torch.from_numpy(weights[3])
net.fc3.weight.data=torch.from_numpy(np.transpose(weights[4]))
net.fc3.bias.data=torch.from_numpy(weights[5])
net.fc4.weight.data=torch.from_numpy(np.transpose(weights[6]))
net.fc4.bias.data=torch.from_numpy(weights[7])


"""
Step 4: test  and save your pytorch model
"""
xReal = np.array([X, Y, Z])
pyPredict0=net.forward(torch.from_numpy(xReal).float())
print(pyPredict0.tolist())


# net.forward([1,2,3])
# pyPredict0=net.forward(torch.from_numpy(np.array([1,2,3])))
