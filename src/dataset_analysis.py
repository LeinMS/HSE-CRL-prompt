import argparse, json, statistics, pathlib
from typing import Dict, Any, List

import numpy as np
from tabulate import tabulate

from src.data_utils import get_dataset_bundle

RESULT_DIR = pathlib.Path("results/dataset_stats")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _percentiles(arr: List[int]):
    arr = np.asarray(arr)
    return {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99)}


def _preview(train_ds, task, tokenizer, num=3):
    rows = []
    for i in range(min(num, len(train_ds))):
        row = train_ds[i]
        text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        if task == "cls":
            rows.append({"text": text+ ("…" if len(text) > 120 else ""),
                          "label": int(row["labels"])})
        else:
            tgt = tokenizer.decode(row["labels"], skip_special_tokens=True)
            rows.append({"src": text[:100] + ("…" if len(text) > 100 else ""),
                          "tgt": tgt[:80] + ("…" if len(tgt) > 80 else "")})
    return rows


def analyse(dataset: str, model: str, max_len: int = 64):
    train, val, n_labels, task, tok = get_dataset_bundle(dataset, model, max_len)
    lengths = [int(mask.sum()) for mask in train["attention_mask"]]
    stats = {
        "dataset": dataset,
        "num_train": len(train),
        "num_val": len(val),
        "avg_len": round(statistics.mean(lengths), 2),
        "max_len": int(max(lengths)),
        **_percentiles(lengths),
        "task": task,
    }
    if task == "cls":
        uniq, counts = np.unique(train["labels"], return_counts=True)
        stats["num_labels"] = int(n_labels)
        stats["class_distribution"] = {int(u): int(c) for u, c in zip(uniq, counts)}

    stats["preview"] = _preview(train, task, tok)

    out_file = RESULT_DIR / f"{dataset}.json"
    out_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _print(stats):
    tbl = [[k, v] for k, v in stats.items() if k not in ("class_distribution", "preview")]
    print(tabulate(tbl, headers=[stats["dataset"], "value"], tablefmt="github"))

    if "class_distribution" in stats:
        print("Class distribution:")
        print(stats["class_distribution"])

    print("First samples:")
    for row in stats["preview"]:
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dataset quick stats & preview")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    info = analyse(args.dataset, args.model, args.max_len)
    _print(info)

