from policy_a2c import PolicyA2C
import torch.optim as optim
import torch

###########################
# Begin hyper-parameters. #
###########################

# Based on A3C paper.
ALPHA = 0.0001
GAMMA = 0.99

USE_GAE = True
LAMBDA = 0.99

GRADIENT_CLIP = 1

#########################
# End hyper-parameters. #
#########################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ControllerA2C():
    def __init__(self, state_size, action_size, rollout_len, seed=1337):
        self.policy = PolicyA2C(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=ALPHA)

        # NOTE: Each entry contains states, actions,
        # rewards, etc. for parallel transitions.
        self.trajectories = []
        self.rollout_len = rollout_len

    def reset(self):
        pass

    def act(self, states):
        states = states.to(device)

        # NOTE: We keep the gradients because the policy outputs
        # are going to be fed back into the policy for learning.
        policy_outputs = self.policy(states)

        return policy_outputs

    def step(self, transitions):
        self.trajectories.append(transitions)

        if len(self.trajectories) < self.rollout_len:
            return

        # Otherwise, it's time to learn.
        self.learn()
        self.trajectories = []

    def learn(self):
        last_transitions = self.trajectories[-1]

        if last_transitions["done"]:
            returns = torch.zeros(self.trajectories[0]["rewards"].size()).to(device)
        else:
            returns = self.policy.bootstrap_values(last_transitions["next_states"].to(device)).detach()

        all_scaled_action_log_probs = []
        all_critic_losses = []

        if USE_GAE:
            all_returns = [returns]

        for transitions in reversed(self.trajectories):
            action_log_probs = transitions["action_log_probs"]
            critic_values = transitions["critic_values"]
            rewards = transitions["rewards"].to(device)

            if USE_GAE:
                advantages = torch.zeros(critic_values.size()).to(device)

                for i in range(len(all_returns)):
                    all_returns[i] = rewards + GAMMA * all_returns[i]
                    advantages += (LAMBDA ** i) * (all_returns[i] - critic_values)

                # NOTE: Although we're supposed to skip this for the
                # last advantage to get the "correct" implementation
                # of GAE, this actually performs better empirically.
                advantages *= 1 - LAMBDA

                all_returns.insert(0, critic_values.detach())
            else:
                returns = rewards + GAMMA * returns
                advantages = returns - critic_values

            # For learning the actor.
            # NOTE: Advantages should be detached when learning the actor
            # since they're purely for scaling action log probabilities.
            scaled_action_log_probs = action_log_probs * advantages.detach()
            all_scaled_action_log_probs.append(scaled_action_log_probs)

            # For learning the critic.
            critic_losses = 0.5 * advantages.pow(2)
            all_critic_losses.append(critic_losses)

        # Negated since it's gradient ascent for policy gradients.
        # By taking the mean across both dimensions of the tensor,
        # we're averaging across steps as well as parallel actors,
        # which translates to averaging across all sampled steps.
        total_actor_loss = -torch.cat(all_scaled_action_log_probs).mean()
        total_critic_loss = torch.cat(all_critic_losses).mean()

        total_loss = total_actor_loss + total_critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

    def get_trajectories_len(self):
        return len(self.trajectories)

    def save(self):
        torch.save(self.policy.cpu().state_dict(), "a2c_policy.pth")
