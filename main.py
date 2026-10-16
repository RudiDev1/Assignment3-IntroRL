import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from train_models import train_model
from test_script import test_script

strategies = ["fixed", "decay_per_episode", "action_gap"]


def train_all():
    for strategy in strategies:
        print("training", strategy)
        train_model(strategy)


def test_all():
    for strategy in strategies:
        print("testing", strategy)
        test_script(strategy + ".pth")


def load_results(strategy):
    df = pd.read_excel(f"experiment_results_{strategy}.pth.xlsx")
    avg_cols = [c for c in df.columns if c.startswith("Avg_")]
    avg_cols = sorted(avg_cols, key=lambda c: float(c.split("_")[1]))
    lengths = [float(c.split("_")[1]) for c in avg_cols]
    avgs = df[avg_cols].iloc[0].values
    stds = df[[c.replace("Avg", "Std") for c in avg_cols]].iloc[0].values
    return lengths, avgs, stds


def compare():
    results = {}
    for strategy in strategies:
        results[strategy] = load_results(strategy)

    x = np.arange(len(results["fixed"][0]))
    width = 0.27
    plt.figure(figsize=(12, 5))
    for i, strategy in enumerate(strategies):
        lengths, avgs, stds = results[strategy]
        plt.bar(x + i * width, avgs, width, yerr=stds, capsize=2, alpha=0.7,
                label=f"{strategy} (avg = {round(np.mean(avgs))})")
    plt.xticks(x + width, [round(l, 2) for l in lengths], rotation=45)
    plt.xlabel("Pole length")
    plt.ylabel("Episode length")
    plt.title("Comparison of the 3 strategies over all pole lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()

    print()
    print("summary")
    for strategy in strategies:
        lengths, avgs, stds = results[strategy]
        print(f"{strategy}: mean = {np.mean(avgs):.1f}, std over lengths = {np.std(avgs):.1f}, "
              f"min = {np.min(avgs):.1f} (length {lengths[np.argmin(avgs)]}), "
              f"max = {np.max(avgs):.1f} (length {lengths[np.argmax(avgs)]})")

    print()
    print("paired t-test over the 30 pole lengths")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a = results[strategies[i]][1]
            b = results[strategies[j]][1]
            t, p = stats.ttest_rel(a, b)
            w, p_w = stats.wilcoxon(a, b)
            print(f"{strategies[i]} vs {strategies[j]}: t = {t:.3f}, p = {p:.4f} | wilcoxon p = {p_w:.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ["train", "test", "compare", "all"]:
        print("usage: python main.py [train|test|compare|all]")
        sys.exit()
    if mode in ["train", "all"]:
        train_all()
    if mode in ["test", "all"]:
        test_all()
    if mode in ["compare", "all"]:
        compare()
