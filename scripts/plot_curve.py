from pathlib import Path
import argparse, json, sys
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 120,
})


def load_jsonl(fname: Path) -> pd.DataFrame:
    if not fname.exists():
        sys.exit(f"[plot_curve] file not found: {fname}")
    with open(fname) as fp:
        rows = [json.loads(line) for line in fp]
    df = pd.DataFrame(rows)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df


def plot_curve(df: pd.DataFrame, metric: str, save_to: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    ax.plot(df["epoch"], df[metric] * 100 if metric.startswith("eval_") else df[metric],
            marker="o", lw=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(save_to)
    print("saved →", save_to)

    if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="experiments/.../<id>")
    ap.add_argument("--metric", default="eval_accuracy", help="колонка из jsonl")
    args = ap.parse_args()

    jsonl = args.run_dir / "epoch_metrics.jsonl"
    df = load_jsonl(jsonl)

    out_png = args.run_dir / f"{args.metric}_curve.png"
    title = str(args.run_dir.relative_to(Path("experiments")))
    plot_curve(df, args.metric, out_png, title)
