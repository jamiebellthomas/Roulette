import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Roulette import Roulette

# --- Training setup ---


# Wrap environment
env = DummyVecEnv([lambda: Roulette()])

# Train model
model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=100_000)

# Test model
test_env = Roulette()
obs, info = test_env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
