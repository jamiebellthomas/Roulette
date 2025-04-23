import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque

class Roulette(gym.Env):
    def __init__(self, initial_bankroll=100, max_steps=100):
        super().__init__()

        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_steps = max_steps
        self.current_step = 0

        self.last_outcome = -1
        self.last_net_reward = 0.0
        self.last_bet_fraction = 0.0

        self.outcomes = deque(maxlen=5000)

        self.bet_options, self.payouts = self._generate_bet_options()
        self.num_bets = len(self.bet_options)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_bets,), dtype=np.float32)

        # Observation = [bankroll, step, last outcome, reward, last bet] + 152 hit rates
        low = np.array([0, 0, 0, -np.inf, 0] + [0.0] * self.num_bets, dtype=np.float32)
        high = np.array([np.inf, max_steps, 36, np.inf, 1] + [1.0] * self.num_bets, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = self.initial_bankroll
        self.current_step = 0
        self.last_outcome = -1
        self.last_net_reward = 0.0
        self.last_bet_fraction = 0.0
        self.outcomes.clear()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        outcome = random.randint(0, 36)
        self.last_outcome = outcome
        self.outcomes.append(outcome)

        total_bet_fraction = np.sum(action)
        total_bet = self.bankroll * total_bet_fraction
        reward = 0

        for i, fraction in enumerate(action):
            amount = self.bankroll * fraction
            if outcome in self.bet_options[i]:
                reward += amount * self.payouts[i]

        net_reward = reward - total_bet
        self.last_net_reward = net_reward
        self.last_bet_fraction = total_bet_fraction

        self.bankroll += net_reward
        self.bankroll = max(self.bankroll, 1e-6)

        roi = (net_reward / total_bet) if total_bet > 0 else 0

        terminated = self.bankroll <= 0.0001
        truncated = self.current_step >= self.max_steps

        info = {
            "outcome": outcome,
            "raw_reward": reward,
            "net_profit": net_reward,
            "total_bet": total_bet,
            "bankroll": self.bankroll
        }

        return self._get_obs(), roi, terminated, truncated, info

    def _get_obs(self):
        hit_rates = self._compute_hit_rates()
        return np.array([
            self.bankroll,
            self.current_step,
            self.last_outcome,
            self.last_net_reward,
            self.last_bet_fraction,
            *hit_rates
        ], dtype=np.float32)

    def _compute_hit_rates(self):
        hit_counts = np.zeros(self.num_bets, dtype=np.int32)
        for outcome in self.outcomes:
            for i, bet_set in enumerate(self.bet_options):
                if outcome in bet_set:
                    hit_counts[i] += 1
        total = len(self.outcomes)
        return hit_counts / total if total > 0 else np.zeros(self.num_bets, dtype=np.float32)

    def render(self, mode="human"):
        print(f"[Step {self.current_step}] Bankroll: {self.bankroll:.2f} | Last outcome: {self.last_outcome} | Last reward: {self.last_net_reward:.2f}")

    def _generate_bet_options(self):
        bet_options = []
        payouts = []

        for n in range(37):  # Straight
            bet_options.append({n})
            payouts.append(36)

        for row in range(12):  # Horizontal splits
            for col in range(2):
                a = 3 * row + col + 1
                bet_options.append({a, a + 1})
                payouts.append(18)

        for row in range(11):  # Vertical splits
            for col in range(3):
                a = 3 * row + col + 1
                bet_options.append({a, a + 3})
                payouts.append(18)

        for row in range(12):  # Street
            start = 3 * row + 1
            bet_options.append({start, start + 1, start + 2})
            payouts.append(12)

        for row in range(11):  # Corner
            for col in range(2):
                a = 3 * row + col + 1
                bet_options.append({a, a + 1, a + 3, a + 4})
                payouts.append(9)

        bet_options.append({0, 1, 2, 3})  # First Four
        payouts.append(9)

        for row in range(11):  # Six Line
            start = 3 * row + 1
            bet_options.append({start, start+1, start+2, start+3, start+4, start+5})
            payouts.append(6)

        bet_options.append(set(range(1, 13)))  # Dozens
        bet_options.append(set(range(13, 25)))
        bet_options.append(set(range(25, 37)))
        payouts.extend([3, 3, 3])

        col1 = set(range(1, 37, 3))  # Columns
        col2 = set(range(2, 37, 3))
        col3 = set(range(3, 37, 3))
        bet_options.extend([col1, col2, col3])
        payouts.extend([3, 3, 3])

        bet_options.append(set(range(1, 19)))  # Low
        bet_options.append(set(range(19, 37)))  # High
        payouts.extend([2, 2])

        red = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
        black = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
        bet_options.extend([red, black])
        payouts.extend([2, 2])

        bet_options.append({i for i in range(1, 37) if i % 2 == 1})  # Odd
        bet_options.append({i for i in range(1, 37) if i % 2 == 0})  # Even
        payouts.extend([2, 2])

        return bet_options, payouts
