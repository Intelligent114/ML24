import math
import json
import random

import gymnasium as gym
import numpy as np
from safetensors.numpy import save_file, load_file
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from utils import StateT, ActionT, RLAlgorithm, Metrics


def valueIteration(
    succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]],
    discount: float,
    epsilon: float = 0.001,
):
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        q = 0.0
        for (next_state, prob, reward) in succAndRewardProb[(state, action)]:
            q += prob * (reward + discount * V[next_state])
        return q

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        policy = {}
        for state in stateActions:
            max_q = -float('inf')
            best_action = None
            for action in stateActions[state]:
                q = computeQ(V, state, action)
                if q > max_q or (q == max_q and action > best_action):
                    max_q = q
                    best_action = action
            if best_action is not None:
                policy[state] = best_action
        return policy

    print("Running valueIteration...")
    V = defaultdict(float)
    numIters = 0
    while True:
        newV = defaultdict(float)
        delta = 0
        for state in stateActions:
            max_q = max(computeQ(V, state, action) for action in stateActions[state])
            newV[state] = max_q
            delta = max(delta, abs(V[state] - newV[state]))
        if delta < epsilon:
            break
        V = newV
        numIters += 1
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)


class ModelBasedMonteCarlo(RLAlgorithm):
    def __init__(
        self,
        actions: List[ActionT],
        discount: float,
        calcValIterEvery: int = 10000,
        explorationProb: float = 0.2,
    ) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        self.tCounts = defaultdict(lambda: defaultdict(int))
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {}

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:

        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:
            explorationProb = 1.0
        elif self.numIters > 1e6:
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        rand_val = random.random()

        if explore and rand_val < explorationProb:
            return random.choice(self.actions)
        else:
            if state in self.pi:
                return self.pi[state]
            else:
                return random.choice(self.actions)

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):

        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            succAndRewardProb = defaultdict(list)

            for (s, a), next_states in self.tCounts.items():
                total_transitions = sum(next_states.values())
                for s_prime, count in next_states.items():
                    prob = count / total_transitions
                    expected_reward = self.rTotal[(s, a)][s_prime] / count
                    succAndRewardProb[(s, a)].append((s_prime, prob, expected_reward))

            self.pi = valueIteration(succAndRewardProb, self.discount, epsilon=0.001)

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "actions": self.actions,
            "discount": self.discount,
            "calcValIterEvery": self.calcValIterEvery,
            "explorationProb": self.explorationProb,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        mcvi_weights = {str(k): np.array(v) for k, v in self.pi.items()}
        save_file(mcvi_weights, path / "mcvi.safetensors")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        pi = load_file(path / "mcvi.safetensors")
        mcvi = cls(**config)
        mcvi.pi = {eval(k): int(v.item()) for k, v in pi.items()}
        return mcvi


class TabularQLearning(RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, explorationProb: float = 0.2, initialQ: float = 0):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.initialQ = initialQ
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:

        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:
            explorationProb = 1.0
        elif self.numIters > 1e5:
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        rand_val = random.random()

        if explore and rand_val < explorationProb:
            return random.choice(self.actions)
        else:
            q_values = [self.Q.get((state, action), self.initialQ) for action in self.actions]
            max_q = max(q_values)
            best_actions = [action for action in self.actions if self.Q.get((state, action), self.initialQ) == max_q]
            return max(best_actions)

    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(
        self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool
    ) -> None:

        gamma = self.discount

        if not hasattr(self, 'state_action_counts'):
            self.state_action_counts = defaultdict(int)

        self.state_action_counts[(state, action)] += 1
        count = self.state_action_counts[(state, action)]

        alpha = 10 * self.getStepSize() / math.log(3 + count)

        current_q = self.Q.get((state, action), self.initialQ)

        if terminal:
            target = reward
        else:
            future_q = max([self.Q.get((nextState, a), self.initialQ) for a in self.actions])
            target = reward + gamma * future_q

        td_error = target - current_q

        self.Q[(state, action)] = current_q + alpha * td_error

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "actions": self.actions,
            "discount": self.discount,
            "explorationProb": self.explorationProb,
            "initialQ": self.initialQ,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        Q_table = {f"{key[0][0]}, {key[0][1]}, {key[1]}": np.array(value) for key, value in self.Q.items()}
        save_file(Q_table, path / "tabular.safetensors")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        rl = cls(**config)
        loaded_data = load_file(path / "tabular.safetensors")
        Q_table = {((eval(key)[0], eval(key)[1]), eval(key)[2]): value.item() for key, value in loaded_data.items()}
        rl.Q = Q_table
        return rl


class Policy(nn.Module, PyTorchModelHubMixin):

    def __init__(self, state_dim: int = 4, action_dim: int = 2, h_size: int = 24):
        super(Policy, self).__init__()

        self.state_projection = nn.Linear(state_dim, h_size)
        self.action_head = nn.Linear(h_size, action_dim)

    def forward(self, x):
        x = F.relu(self.state_projection(x))
        x = self.action_head(x)
        x = F.softmax(x, dim=1)
        return x

    def getAction(self, state: torch.Tensor):
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(
    policy: Policy,
    batch_size: int,
    lr: float,
    num_updates: int,
    max_t: int,
    gamma: float,
    checkpoint_path: Union[str, Path],
    window_size: int = 50,
    save_every: int = 100,
    metrics: Optional[Metrics] = None,
):

    num_params = sum(p.numel() for p in policy.parameters())
    num_params_k = num_params / 10**3
    print(f"Parameters : {num_params_k:.3f}K Total.")
    assert num_params_k < 100, "Your model is too large!"

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()

    R_deque = deque(maxlen=window_size)

    env = gym.make("CartPole-v1")

    status = metrics.get_status() if metrics is not None else "Start training!"
    with tqdm(total=num_updates, desc=status, leave=False) as pbar:
        for upd_step in range(1, num_updates + 1):
            opt.zero_grad()
            for i_episode in range(1, batch_size + 1):
                saved_log_probs = []
                rewards = []
                state, _ = env.reset()

                for t in range(max_t):
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    action, log_prob = policy.getAction(state_tensor)
                    next_state, reward, terminated, truncated, info = env.step(action)

                    saved_log_probs.append(log_prob)
                    rewards.append(reward)

                    state = next_state

                    if terminated:
                        break

                traj_r = sum(rewards)
                R_deque.append(traj_r)

                returns = deque(maxlen=max_t)
                n_steps = len(rewards)

                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.appendleft(G)

                returns_np = np.array(returns)
                returns_mean = returns_np.mean()
                returns_std = returns_np.std() + 1e-8
                returns_normalized = (returns_np - returns_mean) / returns_std
                returns_tensor = torch.tensor(returns_normalized, dtype=torch.float)

                loss = 0
                for log_prob, G_t in zip(saved_log_probs, returns_tensor):
                    loss -= log_prob * G_t
                loss = loss / batch_size

                if metrics is not None:
                    metrics.commit(
                        loss=loss,
                        reward=traj_r,
                    )
                loss.backward()
            opt.step()

            if metrics is not None:
                metrics.commit(update_step_time=True, step=upd_step, lr=lr)
                status = metrics.push()

                if metrics.step % save_every == 0:
                    policy.save_pretrained(checkpoint_path / f"checkpoint_{metrics.step}")
            else:
                status = f"Step: {upd_step}, Average Reward: {np.mean(R_deque):.2f}"

                if upd_step % save_every == 0:
                    policy.save_pretrained(checkpoint_path / f"checkpoint_{upd_step}")

            pbar.update()
            pbar.set_description(status)

    policy.save_pretrained(checkpoint_path / "final")
    print(f"Final Model Saved at {checkpoint_path / 'final'}")

    return policy
