from learn import learn
from plot import run_and_plot
N_STEPS = 1_000_000

def main():
    learn(n_steps=N_STEPS)
    run_and_plot()

if __name__ == "__main__":
    main()