import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot(df: pd.DataFrame, metric: str, save: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    datasets = df['dataset'].unique()

    width = 0.15
    for i, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        x = sub['params'] / 1e3
        y = sub[metric] * 100
        ax.bar(x + i*width, y, width, label=ds)

    ax.set_xlabel("Trainable params, k")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save, dpi=160)
    print("saved →", save)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="eval_accuracy")
    args = ap.parse_args()

    df = pd.read_csv("results/table_all.csv")

    out = Path("results/plots"); out.mkdir(parents=True, exist_ok=True)
    plot(df, args.metric, out / f"{args.metric}.png")
