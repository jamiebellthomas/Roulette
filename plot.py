import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Roulette import Roulette



def run_episode(model, env=None, n_spins=1000):
    if env is None:
        env = Roulette()

    obs, _ = env.reset()
    bankroll_multiplier = 1
    rebuy_amount = env.initial_bankroll

    bet_history = []
    outcome_history = []
    bankroll_history = []

    hit_counts = np.zeros(env.num_bets, dtype=np.int32)

    for _ in range(n_spins):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        outcome = info['outcome']

        # Track which bet options the outcome would have hit
        for i, bet_set in enumerate(env.bet_options):
            if outcome in bet_set:
                hit_counts[i] += 1

        # If bust, rebuy with doubled bankroll
        if terminated:
            bankroll_multiplier *= 2
            rebuy_amount = env.initial_bankroll * bankroll_multiplier
            env.bankroll = rebuy_amount
            env.last_outcome = -1
            env.last_net_reward = 0.0
            env.last_bet_fraction = 0.0
            obs = env._get_obs()

        bet_history.append(action.flatten())
        outcome_history.append(outcome)
        bankroll_history.append(obs[0])

    hit_frequencies = hit_counts / n_spins

    return {
        "bet_history": np.array(bet_history),
        "outcome_history": np.array(outcome_history),
        "bankroll_history": np.array(bankroll_history),
        "hit_frequencies": hit_frequencies
    }






def plot_average_bets(bet_history):
    mean_bets = bet_history.mean(axis=0)
    labels = generate_bet_labels()
    print("Sum:", sum(mean_bets))
    plt.figure(figsize=(16, 6))
    plt.bar(labels, mean_bets)
    plt.title("Average Bet Proportions Per Bet Option")
    plt.xlabel("Bet Option")
    plt.ylabel("Average Proportion of Bankroll Bet")
    plt.xticks(rotation=90)  # <-- rotate x labels
    plt.grid(axis='y')
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




def generate_bet_labels():
    labels = []

    # Straight bets (0–36)
    for n in range(37):
        labels.append(f"Straight {n}")

    # Split (horizontal)
    for row in range(12):
        for col in range(2):
            a = 3 * row + col + 1
            labels.append(f"Split {a}-{a+1}")

    # Split (vertical)
    for row in range(11):
        for col in range(3):
            a = 3 * row + col + 1
            labels.append(f"Split {a}-{a+3}")

    # Street (3 in a row)
    for row in range(12):
        start = 3 * row + 1
        labels.append(f"Street {start}-{start+2}")

    # Corner
    for row in range(11):
        for col in range(2):
            a = 3 * row + col + 1
            labels.append(f"Corner {a},{a+1},{a+3},{a+4}")

    # First four
    labels.append("First Four (0,1,2,3)")

    # Six line
    for row in range(11):
        start = 3 * row + 1
        labels.append(f"Six Line {start}-{start+5}")

    # Dozens
    labels.append("Dozen 1 (1–12)")
    labels.append("Dozen 2 (13–24)")
    labels.append("Dozen 3 (25–36)")

    # Columns
    labels.append("Column 1")
    labels.append("Column 2")
    labels.append("Column 3")

    # Low/High
    labels.append("Low (1–18)")
    labels.append("High (19–36)")

    # Red/Black
    labels.append("Red")
    labels.append("Black")

    # Odd/Even
    labels.append("Odd")
    labels.append("Even")

    return labels

def count_hits_from_outcomes(outcomes, bet_options):
    """
    Count how many times each bet option would have hit based on the outcome history.

    Args:
        outcomes (list[int] or np.ndarray): Sequence of spin outcomes (0–36).
        bet_options (list[set[int]]): Each set contains numbers that win for that bet type.

    Returns:
        np.ndarray: Integer hit counts per bet option (length = len(bet_options)).
    """
    hit_counts = np.zeros(len(bet_options), dtype=np.int32)

    for outcome in outcomes:
        for i, bet_set in enumerate(bet_options):
            if outcome in bet_set:
                hit_counts[i] += 1

    return hit_counts

def plot_hit_vs_selected(hit_counts, bet_history, bet_labels=None, top_n=20):
    """
    Compare hit rate vs selection rate for each bet option, sorted by hit rate.

    Args:
        hit_counts (np.ndarray): Number of hits per option (e.g. from count_hits_from_outcomes).
        bet_history (np.ndarray): Shape (n_steps, n_bets), showing proportion bet on each option.
        bet_labels (list[str], optional): Human-readable labels.
        top_n (int): Show top N options sorted by hit rate.
    """
    n_spins = len(bet_history)
    total_bet_per_option = bet_history.sum(axis=0)
    selection_rate = total_bet_per_option / n_spins
    hit_rate = hit_counts / n_spins

    # Sort by hit rate
    sorted_indices = np.argsort(hit_rate)[-top_n:][::-1]

    hits = hit_rate[sorted_indices]
    selected = selection_rate[sorted_indices]
    labels = (
        [f"Bet {i}" for i in sorted_indices] if bet_labels is None
        else [bet_labels[i] for i in sorted_indices]
    )

    x = np.arange(top_n)
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, hits, width, label="Hit Rate")
    plt.bar(x + width/2, selected, width, label="Selection Rate")

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Rate (per spin)")
    plt.title("Hit Rate vs Selection Rate (Sorted by Hit Rate)")
    plt.legend()
    plt.grid(axis='y')
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


    hits = count_hits_from_outcomes(results["outcome_history"], env.bet_options)
    labels = generate_bet_labels()

    plot_hit_vs_selected(hits, results["bet_history"], bet_labels=labels, top_n=20)



if __name__ == "__main__":
    run_and_plot()

