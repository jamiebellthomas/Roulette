import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn.functional as F

class Roulette(gym.Env):
    def __init__(self, initial_bankroll=100, max_steps=100):
        super().__init__()

        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_steps = max_steps
        self.current_step = 0

        # Generate all betting options and their payout multipliers
        self.bet_options, self.payouts = self._generate_bet_options()
        self.num_bets = len(self.bet_options)

        # Action: continuous proportions per bet (∈ [0,1], sum ≤ 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_bets,), dtype=np.float32)

        # Observation: current bankroll
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = self.initial_bankroll
        self.current_step = 0
        observation = np.array([self.bankroll], dtype=np.float32)
        info = {}  # optional metadata
        return observation, info
    
    def step(self, action):
        self.current_step += 1
        action = F.softmax(torch.tensor(action), dim=0).numpy()


        # Normalize bets to total ≤ 1.0
        total_bet_fraction = np.sum(action)
        if total_bet_fraction > 1.0:
            action = action / total_bet_fraction

        outcome = random.randint(0, 36)
        total_bet = self.bankroll * np.sum(action)
        reward = 0

        for i, fraction in enumerate(action):
            amount = self.bankroll * fraction
            if outcome in self.bet_options[i]:
                reward += amount * self.payouts[i]

        net_reward = reward - total_bet
        self.bankroll += net_reward
        terminated = self.bankroll <= 0.0001         # Bankroll ran out
        truncated = self.current_step >= self.max_steps  # Hit max steps
        info = {"outcome": outcome, "raw_reward": reward, "net_profit": net_reward}

        return np.array([self.bankroll], dtype=np.float32), net_reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Bankroll: {self.bankroll:.2f}")

    def _generate_bet_options(self):
        bet_options = []
        payouts = []

        # Straight up (0–36)
        for n in range(37):
            bet_options.append({n})
            payouts.append(36)

        # Split (horizontal + vertical)
        for row in range(12):
            for col in range(2):  # Horizontal split
                a = 3 * row + col + 1
                bet_options.append({a, a + 1})
                payouts.append(18)
        for row in range(11):
            for col in range(3):  # Vertical split
                a = 3 * row + col + 1
                bet_options.append({a, a + 3})
                payouts.append(18)

        # Street (3 numbers in a row)
        for row in range(12):
            start = 3 * row + 1
            bet_options.append({start, start + 1, start + 2})
            payouts.append(12)

        # Corner
        for row in range(11):
            for col in range(2):
                a = 3 * row + col + 1
                bet_options.append({a, a + 1, a + 3, a + 4})
                payouts.append(9)

        # First four (0,1,2,3)
        bet_options.append({0, 1, 2, 3})
        payouts.append(9)

        # Six line (two adjacent rows)
        for row in range(11):
            start = 3 * row + 1
            bet_options.append({start, start+1, start+2, start+3, start+4, start+5})
            payouts.append(6)

        # Dozens
        bet_options.append(set(range(1, 13)))     # 1–12
        bet_options.append(set(range(13, 25)))    # 13–24
        bet_options.append(set(range(25, 37)))    # 25–36
        payouts.extend([3, 3, 3])

        # Columns
        col1 = set(range(1, 37, 3))
        col2 = set(range(2, 37, 3))
        col3 = set(range(3, 37, 3))
        bet_options.extend([col1, col2, col3])
        payouts.extend([3, 3, 3])

        # Low/High
        bet_options.append(set(range(1, 19)))    # Low
        bet_options.append(set(range(19, 37)))   # High
        payouts.extend([2, 2])

        # Red/Black
        red = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
        black = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
        bet_options.extend([red, black])
        payouts.extend([2, 2])

        # Odd/Even
        bet_options.append({i for i in range(1, 37) if i % 2 == 1})  # Odd
        bet_options.append({i for i in range(1, 37) if i % 2 == 0})  # Even
        payouts.extend([2, 2])

        return bet_options, payouts
    

if __name__ == "__main__":
    env = Roulette()
    obs = env.reset()
    done = False

    while not done:
        action = np.random.dirichlet(np.ones(env.num_bets))  # Random legal proportions
        obs, reward, done, info = env.step(action)
        env.render()
