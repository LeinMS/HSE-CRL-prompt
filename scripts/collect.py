import json, argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def collect(root: Path = Path("experiments")) -> pd.DataFrame:
    records = []
    for mfile in root.glob("*/*/metrics.json"):
        data = json.loads(mfile.read_text())
        records.append(data)
    df = pd.DataFrame(records)
    df = df.sort_values(["dataset", "method"])
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df = collect()
    outdir = Path("results"); outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "table_all.csv", index=False)
    df.to_json(outdir / "table_all.json", orient="records", indent=2)

    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))
