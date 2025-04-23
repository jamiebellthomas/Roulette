import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Roulette import Roulette
from CustomActorCritic import CustomActorCritic
N_STEPS = 100_000
# --- Training setup ---

def learn(n_steps):
    # Wrap environment
    env = DummyVecEnv([lambda: Roulette()])

    # Train model
    model = PPO(CustomActorCritic, env, verbose=1, device='cpu')
    model.learn(total_timesteps=n_steps)

    # Save model
    model.save("ppo_roulette")

if __name__ == "__main__":
    learn(n_steps=N_STEPS)
