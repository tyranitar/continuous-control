from policy_d4pg import ActorNet, CriticNet
from collections import deque
import torch.optim as optim
from random import sample
import random
import torch

###########################
# Begin hyper-parameters. #
###########################

ACTOR_LR = 0.001
CRITIC_LR = 0.001

REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 64

GAMMA = 0.99
TAU = 0.001

GRADIENT_CLIP = 1

#########################
# End hyper-parameters. #
#########################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ControllerD4PG():
    def __init__(self, state_size, action_size, rollout_len, seed=1337):
        random.seed(seed)

        self.n_atoms = 51
        # v_min = sum([-0.1 * (0.99 ** i) for i in range(1000)])
        # since the min reward at each step is -0.1 after reward
        # shaping, gamma = 0.99, and each episode has 1000 steps.
        self.v_min = -10
        self.v_max = 0
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)

        self.actor = ActorNet(state_size, action_size, seed).to(device)
        self.critic = CriticNet(state_size, action_size, self.n_atoms, seed).to(device)

        self.target_actor = ActorNet(state_size, action_size, seed).to(device)
        self.target_critic = CriticNet(state_size, action_size, self.n_atoms, seed).to(device)

        # Initialize target networks.
        self.soft_update_target_nets(1)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.trajectories = deque(maxlen=rollout_len)
        self.rollout_len = rollout_len

    def reset(self):
        self.trajectories = deque(maxlen=self.rollout_len)

    def act(self, states):
        states = states.to(device)

        with self.actor.eval_no_grad():
            actions = self.actor(states)

        return {
            "actions": actions,
        }

    def step(self, transitions):
        # NOTE: Shape rewards to improve convergence. This
        # will shift rewards from {0, 0.1} to {-0.1, 0}.
        transitions["rewards"] -= 0.1

        exp_tuples = self.convert_to_exp_tuples(transitions)
        self.trajectories.append(exp_tuples)

        if len(self.trajectories) < self.rollout_len:
            return

        self.replay_buffer += zip(*self.trajectories)

        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Otherwise, it's time to learn.
        self.learn()

    def learn(self):
        samples = sample(self.replay_buffer, BATCH_SIZE)
        trajectories = [self.convert_to_transitions(exp_tuples) for exp_tuples in zip(*samples)]

        first_transitions = trajectories[0]
        first_states = first_transitions["states"]
        first_actions = first_transitions["actions"]

        local_q_dists = self.critic(first_states, first_actions)
        projected_target_q_dists = self.get_projected_target_q_dists(trajectories)

        # This is equivalent to the DQN loss.
        # The negative here is for cross-entropy loss.
        total_critic_loss = -(torch.log(local_q_dists + 1e-10) * projected_target_q_dists).sum(dim=-1).mean()

        # NOTE: This will zero out the critic
        # grad from updating the actor below.
        self.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRADIENT_CLIP)
        self.critic_opt.step()

        actions = self.actor(first_states)
        q_dists = self.critic(first_states, actions)

        # Negative since this is gradient ascent.
        total_actor_loss = -q_dists.matmul(self.v_lin).mean()

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRADIENT_CLIP)
        self.actor_opt.step()

        self.soft_update_target_nets(TAU)

    def soft_update_target_nets(self, tau):
        self.soft_update_target_net(self.actor, self.target_actor, tau)
        self.soft_update_target_net(self.critic, self.target_critic, tau)

    def soft_update_target_net(self, local_net, target_net, tau):
        for target_params, local_params in zip(
            target_net.parameters(),
            local_net.parameters(),
        ):
            target_params.data.copy_(
                tau * local_params.data + \
                (1 - tau) * target_params.data
            )

    def convert_to_exp_tuples(self, transitions):
        # NOTE: This must return a list instead of a zip
        # object since we need to reuse it multiple times
        # when adding to the replay buffer. A zip object
        # is a generator and hence can only be used once.
        return list(zip(
            transitions["states"].cpu(),
            transitions["actions"].cpu(),
            transitions["rewards"].cpu(),
            transitions["next_states"].cpu(),
        ))

    def convert_to_transitions(self, exp_tuples):
        states, actions, rewards, next_states = zip(*exp_tuples)

        return {
            "states": torch.stack(states).to(device),
            "actions": torch.stack(actions).to(device),
            "rewards": torch.stack(rewards).to(device),
            "next_states": torch.stack(next_states).to(device),
        }

    def get_projected_target_q_dists(self, trajectories):
        N = len(trajectories)

        last_transitions = trajectories[-1]
        last_next_states = last_transitions["next_states"]

        discounted_rewards = torch.zeros(last_transitions["rewards"].size()).to(device)

        for transitions in reversed(trajectories):
            discounted_rewards = transitions["rewards"] + GAMMA * discounted_rewards

        # Since reward tensors have shape (batch size, 1).
        discounted_rewards = discounted_rewards.squeeze()

        target_actions = self.target_actor(last_next_states)
        target_q_dists = self.target_critic(last_next_states, target_actions)

        projected_target_q_dists = torch.zeros(target_q_dists.size()).to(device)

        for j in range(self.n_atoms):
            Tz_j = torch.clamp(
                discounted_rewards + (GAMMA ** N) * (self.v_min + j * self.delta),
                min=self.v_min,
                max=self.v_max,
            )

            b_j = (Tz_j - self.v_min) / self.delta
            l = b_j.floor().long()
            u = b_j.ceil().long()

            eq_mask = l == u
            ne_mask = l != u

            # If `a` is a 3 x 3 tensor, then `a[[0, 1], [1, 2]]`
            # gets the 1st and 2nd rows of the tensor, and then
            # respectively the 2nd and 3rd columns of those rows.
            projected_target_q_dists[eq_mask, l[eq_mask]] += target_q_dists[eq_mask, j]

            # NOTE: We use `u - b_j` for the floor atoms since if
            # `b_j` is closer to `l`, this will yield a higher %
            # of the target probability to `l` and vice versa. It
            # feels counterintuitive at first but it makes sense.
            projected_target_q_dists[ne_mask, l[ne_mask]] += target_q_dists[ne_mask, j] * (u.float() - b_j)[ne_mask]
            projected_target_q_dists[ne_mask, u[ne_mask]] += target_q_dists[ne_mask, j] * (b_j - l.float())[ne_mask]

        return projected_target_q_dists.detach()

    def get_trajectories_len(self):
        # Trajectories are reset in the `reset` method.
        return 0

    def save(self):
        torch.save(self.actor.cpu().state_dict(), "d4pg_actor.pth")
        torch.save(self.critic.cpu().state_dict(), "d4pg_critic.pth")
