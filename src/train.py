from collections import deque
from time import time
import numpy as np
import torch
import gc

NUM_EPISODES = 1000

# NOTE: `T_MAX` should be an integer multiple of `ROLLOUT_LEN`.
ROLLOUT_LEN = 5
T_MAX = 1000

SCORES_WINDOW_SIZE = 100
TARGET_SCORE = 30

def train(
    env,
    create_controller,
    brain_name,
    num_agents,
    num_episodes=NUM_EPISODES,
):
    gc.disable()

    scores_window = deque(maxlen=SCORES_WINDOW_SIZE)
    scores_history = []
    last_avg_score = 0

    controller = create_controller(ROLLOUT_LEN)

    for i in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        controller.reset()

        states = torch.from_numpy(env_info.vector_observations).float()
        scores = np.zeros(num_agents)

        start_time = time()

        for _ in range(T_MAX):
            outputs = controller.act(states)
            actions = outputs["actions"].cpu().numpy()
            env_info = env.step(actions)[brain_name]

            next_states = torch.from_numpy(env_info.vector_observations).float()
            rewards = env_info.rewards
            dones = env_info.local_done

            assert np.all(dones) or not np.any(dones), \
                "Expected all or none to be done."

            controller.step({
                "states": states,
                # The `rewards` output is a list for some reason.
                "rewards": torch.tensor(rewards).float().unsqueeze(-1),
                "next_states": next_states,
                "done": np.any(dones),
                # Feed controller output back to itself.
                **outputs,
            })

            states = next_states
            scores += rewards

        elapsed_time = time() - start_time

        # Instead of throwing, we can also flush the controller.
        assert controller.get_trajectories_len() == 0, \
            "Expected controller to have empty trajectories."

        # For this episode.
        ep_score = np.mean(scores)

        scores_window.append(ep_score)
        scores_history.append(ep_score)

        # For the last 100 episodes.
        avg_score = np.mean(scores_window)

        avg_score_delta = avg_score - last_avg_score
        last_avg_score = avg_score

        gc.collect()

        print(f"\rep {i}\tscore {ep_score:.2f}\tavg {avg_score:.2f}\tdelta {avg_score_delta:.2f}\ttime {elapsed_time:.2f}s\t", end="")

        if i % 10 == 0:
            print(f"\rep {i}\tscore {ep_score:.2f}\tavg {avg_score:.2f}\tdelta {avg_score_delta:.2f}\ttime {elapsed_time:.2f}s\t")

        if i >= 50 and avg_score < 2:
            raise Exception("Training aborted due to slow learning.")

        if i >= SCORES_WINDOW_SIZE and avg_score >= TARGET_SCORE:
            print(f"\nenv solved in {i - SCORES_WINDOW_SIZE} episodes!\tavg score: {avg_score:.2f}")

            controller.save()

            break

    gc.enable()

    return scores_history
