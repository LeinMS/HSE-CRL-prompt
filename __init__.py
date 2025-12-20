import os
import shutil
from pathlib import Path
import argparse

def main(src_root: str, dest_dir: str):
    src_root = Path(src_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for a in src_root.iterdir():
        if not a.is_dir():
            continue
        if a.name == 'roberta-base' or a.name == 'tiiuae':
            if a.name == 'tiiuae':
                a = a / 'falcon-rw-1b'
            for b in a.iterdir():
                if b.name in ['sst2', 'ag_news', "trec"]:
                    if not b.is_dir():
                        continue
                    for c in b.iterdir():
                        if not c.is_dir():
                            continue
                        src_file = c / 'epoch_metrics.jsonl'
                        if src_file.is_file():
                            new_name = f"{a.name}_{b.name}_{c.name}_metrics.jsonl"
                            dest_file = dest_dir / new_name
                            shutil.copy2(src_file, dest_file)
                            print(f"Copied {src_file} -> {dest_file}")
    for a in src_root.iterdir():
        if not a.is_dir():
            continue
        if a.name == 'roberta-base' or a.name == 'tiiuae':
            if a.name == 'tiiuae':
                a = a / 'falcon-rw-1b'
            for b in a.iterdir():
                if b.name in ['sst2', 'ag_news', "trec"]:
                    if not b.is_dir():
                        continue
                    for c in b.iterdir():
                        if not c.is_dir():
                            continue
                        if c.name == 'ape':
                            src_file = c / 'metrics.json'
                            if src_file.is_file():
                                new_name = f"{a.name}_{b.name}_{c.name}_metrics.json"
                                dest_file = dest_dir / new_name
                                shutil.copy2(src_file, dest_file)
                                print(f"Copied {src_file} -> {dest_file}")

main("./experiments", "./RES")