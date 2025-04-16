import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Roulette import Roulette



def run_episode(model, env=None):
    if env is None:
        env = Roulette()

    obs, _ = env.reset()
    done = False

    bet_history = []
    outcome_history = []
    bankroll_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        assert np.isclose(action.sum(), 1.0, atol=1e-5), f"Bad action sum: {action.sum()}"
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        bet_history.append(action.flatten())
        outcome_history.append(info['outcome'])
        bankroll_history.append(obs[0])

    return {
        "bet_history": np.array(bet_history),
        "outcome_history": np.array(outcome_history),
        "bankroll_history": np.array(bankroll_history)
    }




def plot_average_bets(bet_history):
    mean_bets = bet_history.mean(axis=0)
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(mean_bets)), mean_bets)
    plt.title("Average Bet Proportions Per Bet Option")
    plt.xlabel("Bet Option Index (0–151)")
    plt.ylabel("Average Proportion of Bankroll Bet")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bankroll_over_time(bankroll_history):
    plt.figure(figsize=(10, 4))
    plt.plot(bankroll_history)
    plt.title("Bankroll Over Time")
    plt.xlabel("Step")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_outcomes(outcome_history):
    plt.figure(figsize=(10, 4))
    plt.hist(outcome_history, bins=np.arange(38) - 0.5, edgecolor='black')
    plt.title("Distribution of Outcomes (Spin Results)")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(37))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_and_plot():

    # Re-create the environment
    env = Roulette()

    # Load the model
    model = PPO.load("ppo_roulette", env=env)

    results = run_episode(model=model, env=env)

    # Plot everything
    plot_average_bets(results["bet_history"])
    plot_bankroll_over_time(results["bankroll_history"])
    plot_outcomes(results["outcome_history"])


if __name__ == "__main__":
    run_and_plot()

