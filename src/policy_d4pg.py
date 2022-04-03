from contextlib import contextmanager
import torch.nn.functional as F
from nn_utils import init_orth
import torch.nn as nn
import torch

# States: 33
# Actions: 4

init_fc = init_orth

def get_fc_layer(in_dim, out_dim):
    return nn.Sequential(
        init_fc(nn.Linear(in_dim, out_dim)),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )

class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, seed=1337):
        super(ActorNet, self).__init__()
        torch.manual_seed(seed)

        fc_layer_dims = [
            state_size,
            400,
            300,
        ]

        self.net = nn.Sequential(
            *self.get_fc_layers(fc_layer_dims),
            init_fc(nn.Linear(fc_layer_dims[-1], action_size), 1e-3),
            nn.Tanh(),
        )

    def forward(self, states):
        return self.net(states)

    def get_fc_layers(self, layer_dims):
        return [
            get_fc_layer(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ]

    @contextmanager
    def eval_no_grad(self):
        with torch.no_grad():
            try:
                self.eval()
                yield
            finally:
                self.train()

class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, n_atoms, seed=1337):
        super(CriticNet, self).__init__()
        torch.manual_seed(seed)

        layer_dims = [
            state_size,
            400,
            300,
        ]

        self.fc1 = get_fc_layer(layer_dims[0], layer_dims[1])
        self.fc2 = get_fc_layer(layer_dims[1] + action_size, layer_dims[2])
        self.fc3 = init_fc(nn.Linear(layer_dims[-1], n_atoms))

    def forward(self, states, actions):
        x = states
        a = actions
        x = self.fc1(x)
        x = torch.cat([x, a], dim=-1)
        x = self.fc2(x)

        return F.softmax(self.fc3(x), dim=-1)
