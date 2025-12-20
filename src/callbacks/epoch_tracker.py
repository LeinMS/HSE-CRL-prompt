from pathlib import Path
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json, time

class EpochTracker(TrainerCallback):
    def __init__(self, save_dir: Path):
        self.fname = Path(save_dir) / "epoch_metrics.jsonl"
        self.start = time.time()
        self.seen  = set()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        key = (state.epoch, state.global_step)
        if key in self.seen:
            return
        self.seen.add(key)
        row = dict(
            seconds = round(time.time() - self.start, 1),
            **metrics,
        )
        with open(self.fname, "a") as fp:
            fp.write(json.dumps(row) + "\n")
