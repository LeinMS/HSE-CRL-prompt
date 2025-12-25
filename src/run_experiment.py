import importlib, json, time, os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer
import evaluate
from src.data_utils import get_dataset_bundle, get_data_collator
from src.callbacks.epoch_tracker import EpochTracker

_METHODS = {
    "prompt_v1": "src.methods.prompt_v1.PromptV1Runner",
    "prompt_v2": "src.methods.prompt_v2.PromptV2Runner",
    "mixture_soft": "src.methods.mixture_soft.MixtureSoftRunner",
    "textgrad": "src.methods.textgrad.TextGradPromptRunner",
    "cons_prompt": "src.methods.consprompt_runner.ConsPromptRunner",
    "prewrite_rl": "src.methods.prewrite_rl.PrewriteRlRunner",
    "ape": "src.methods.ape.APERunner",
    "ptcprl_embedding": "src.methods.ptcprl_embedding_runner.PTCPRLEmbeddingRunner",
}


def _load_runner(name: str):
    if name not in _METHODS:
        raise ValueError(f"Unknown method '{name}'")
    module, cls_name = _METHODS[name].rsplit(".", 1)
    return getattr(importlib.import_module(module), cls_name)


@hydra.main(version_base=None, config_path="../configs", config_name="")
def main(cfg: DictConfig):
    train, val, test_ds, n_labels, task, tokenizer, train_raw, val_raw = get_dataset_bundle(
        cfg.dataset, cfg.model.name, cfg.get("max_len", 128)
    )
    collator = get_data_collator(task, tokenizer)
    
    Runner = _load_runner(cfg.method)
    runner = Runner.build(cfg.model.name, task, **cfg.method_cfg, device=cfg.device, num_labels=n_labels, tokenizer=tokenizer)
    
    if cfg.method in ["gptswarm", "prewrite_rl", "textgrad", "ape"]:
        train_raw = train_raw.select(range(100))
        val_raw = val_raw.select(range(100))
        runner.search(train_raw, val_raw)

        if cfg.method in ["textgrad", "ape"]:
            metrics_val = {}
            if cfg.method == "textgrad":
                metrics_val = trainer.evaluate(val)
            else:
                metrics_val = {
                    "eval_accuracy": runner.best_prompt.score if runner.best_prompt else 0.0,
                    "best_prompt": runner.best_prompt.template if runner.best_prompt else ""
                }
            
            metrics = {
                "dataset": cfg.dataset,
                "method": cfg.method,
                "epochs": 0,
                **metrics_val,
            }
            print(metrics)
            out_dir = Path("experiments") / cfg.model.name / cfg.dataset / cfg.method
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            
            print(f"\nMetrics saved to {out_dir / 'metrics.json'}")
            print(f"Best prompt: {metrics['best_prompt']}")
            print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
            return
    
    model = runner.model

    if task == "cls":
        metric_acc = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            return metric_acc.compute(predictions=preds, references=labels)
    else:
        compute_metrics = None

    out_dir = Path("experiments") / cfg.model.name / cfg.dataset / cfg.method
    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy" if task == "cls" else None,
        greater_is_better=True,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch,
        per_device_eval_batch_size=cfg.train.batch,
        learning_rate=cfg.train.lr,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_safetensors=False,
        eval_accumulation_steps=10,
    )
    tracker = EpochTracker(out_dir)
    if cfg.method == "ptcprl_embedding":
        trainer = runner.build_trainer(train, val, collator, args, compute_metrics)
        trainer.add_callback(tracker)
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train,
            eval_dataset=val,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[tracker],
        )

    start = time.time()
    trainer.train()
    print(trainer.evaluate(test_ds)['eval_accuracy'])
    elapsed = round(time.time() - start, 2)

    metrics = {
        "dataset": cfg.dataset,
        "method": cfg.method,
        "train_time_sec": elapsed,
        "epochs": cfg.train.epochs,
    }
    if task == "cls":
        metrics.update(trainer.evaluate(val))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    art_dir = Path("artifacts") / f"{cfg.dataset}_{cfg.method}_best"
    art_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(art_dir)

    print("Finished experiment →", art_dir)


if __name__ == "__main__":
    main()
