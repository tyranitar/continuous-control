from contextlib import contextmanager
import torch.nn.functional as F
from nn_utils import init_orth
import torch.nn as nn
import torch

# States: 33
# Actions: 4

class PolicyA2C(nn.Module):
    def __init__(self, state_size, action_size, seed=1337):
        super(PolicyA2C, self).__init__()
        torch.manual_seed(seed)

        actor_layer_dims = [
            state_size,
            128,
            64,
        ]

        self.actor_net = nn.Sequential(
            *self.get_fc_layers(actor_layer_dims),
            init_orth(nn.Linear(actor_layer_dims[-1], action_size)),
        )

        self.actor_stds = nn.Parameter(torch.zeros(action_size))

        critic_layer_dims = [
            state_size,
            128,
            64,
        ]

        self.critic_net = nn.Sequential(
            *self.get_fc_layers(critic_layer_dims),
            init_orth(nn.Linear(critic_layer_dims[-1], 1)),
        )

    def forward(self, states):
        actor_outputs = self.actor_net(states)

        means = F.tanh(actor_outputs)
        stds = F.softplus(self.actor_stds)

        action_dists = torch.distributions.Normal(means, stds)
        actions = action_dists.sample()

        # NOTE: Combine individual action probabilities by summing log
        # probabilities. The result is a vector of log probabilities,
        # one for each parallel agent's action in the batch.
        #
        # The log probabilities of individual action components are summed
        # to compute the log probability of the overall action vector. The
        # product of independent probabilities that make up the overall
        # action's probability becomes a sum of log probabilities.
        #
        # The probabilities of individual action components are independent
        # b/c they're not causally related by virtue of the NN architecture.
        action_log_probs = action_dists.log_prob(actions).sum(-1).unsqueeze(-1)

        # NOTE: This critic doesn't take in the actor's actions as input. It
        # indirectly influences the actor by setting a baseline for the SVF.
        critic_values = self.critic_net(states)

        return {
            "actions": actions,
            "action_log_probs": action_log_probs,
            "critic_values": critic_values,
        }

    # For bootstrapping the return of
    # the last state in the trajectory.
    def bootstrap_values(self, states):
        with self.eval_no_grad():
            ret = self.critic_net(states)

        return ret

    def get_fc_layers(self, layer_dims):
        return [
            nn.Sequential(
                init_orth(nn.Linear(layer_dims[i], layer_dims[i + 1])),
                nn.ReLU(),
            )
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
