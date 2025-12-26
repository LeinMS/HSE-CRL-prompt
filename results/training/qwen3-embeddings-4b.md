# PTCPRL Experiment Report (Qwen/Qwen3-Embedding-4B on SST-2)

This README documents the PTCPRL (contrastive + RL) soft-prompt experiment run on the SST-2 dataset using the Qwen/Qwen3-Embedding-4B model. It includes the exact command used, configuration details, training resources, and the resulting metrics.

## Summary

**Goal:** Run the author’s PTCPRL-style algorithm (MLE + contrastive + RL prompt optimization) on a modern embedding model for an NLU benchmark.

**Model:** `Qwen/Qwen3-Embedding-4B`

**Dataset:** SST-2 (GLUE) sentiment classification

**Method:** `ptcprl_embedding` (TwoPhaseTrainer: MLE → MLE+contrastive+RL)

## What Was Trained

We trained **soft prompt embeddings** (virtual tokens) while keeping the base model frozen. The model prepends trainable prompt vectors to each input, and the Trainer optimizes the prompt via:

1. **Phase 1 (MLE):** Cross-entropy loss only.
2. **Phase 2 (PTCPRL):** Combined loss
   - MLE loss (alpha)
   - Contrastive loss on mutated prompts (beta)
   - RL-style reward (gamma)

In this run, we trained for **2 epochs**, which means only the MLE phase is active (the PTCPRL phase begins after `phase1_epochs=3`).

## Exact Command Used

```bash
!PYTHONPATH=. {sys.executable} -m src.run_experiment \
  --config-name sst2_ptcprl_embedding_qwen3_4b \
  train.batch=4 \
  train.epochs=2 \
  method_cfg.num_virtual_tokens=10 \
  method_cfg.rl_subset_size=8 \
  method_cfg.k_negatives=1
```

## Configuration Overrides

The run used the YAML config `configs/configs/sst2_ptcprl_embedding_qwen3_4b.yaml` and applied the overrides above.

| Parameter | Value | Notes |
|---|---:|---|
| `train.batch` | 4 | Smaller batch for GPU memory stability |
| `train.epochs` | 2 | Short run (only MLE phase active) |
| `method_cfg.num_virtual_tokens` | 10 | Prompt length (trainable tokens) |
| `method_cfg.rl_subset_size` | 8 | RL reward subset size |
| `method_cfg.k_negatives` | 1 | Negative prompt samples |

## Resources Used

- **GPU:** CUDA (Colab GPU)
- **Model size:** ~8 GB checkpoints (2 shards)
- **Training time:** ~3206 sec (≈53 min)

## Training Metrics

### Per-epoch validation results (epoch_metrics.jsonl)

| Epoch | Eval Loss | Eval Accuracy | Eval Runtime (s) |
|---:|---:|---:|---:|
| 1 | 0.5920 | 0.9232 | 85.37 |
| 2 | 0.2316 | 0.9550 | 85.36 |

### Final metrics (metrics.json)

| Metric | Value |
|---|---:|
| `train_time_sec` | 3216.72 |
| `epochs` | 2 |
| `eval_loss` | 0.2316 |
| `eval_accuracy` | 0.9550 |
| `eval_samples_per_second` | 78.764 |
| `eval_steps_per_second` | 19.694 |

## Training Process Analysis

- Accuracy improved **from 0.923 → 0.955** between epoch 1 and 2.
- Loss decreased **from 0.592 → 0.232**, indicating stable learning.
- The run used only the **MLE phase** since `phase1_epochs=3` and total epochs were 2.
- For full PTCPRL behavior (contrastive + RL), the run should be extended to **≥3 epochs**.

## Output Artifacts and Logs

- **Epoch logs:** `experiments/<model>/<dataset>/<method>/epoch_metrics.jsonl`
- **Final metrics:** `experiments/<model>/<dataset>/<method>/metrics.json`
- **Best model checkpoint:** `artifacts/sst2_ptcprl_embedding_best/`

## Reproducibility Notes

To reproduce the same setup, run the command above. If you want to enable the full PTCPRL phase, increase epochs to 3 or more:

```bash
!PYTHONPATH=. {sys.executable} -m src.run_experiment \
  --config-name sst2_ptcprl_embedding_qwen3_4b \
  train.batch=4 \
  train.epochs=3 \
  method_cfg.num_virtual_tokens=10 \
  method_cfg.rl_subset_size=8 \
  method_cfg.k_negatives=1
```

## Appendix: Key Files

- `src/methods/ptcprl_embedding_runner.py` — PTCPRL implementation
- `src/run_experiment.py` — Hydra entry point + Trainer selection
- `src/data_utils.py` — dataset loading and tokenization
- `configs/configs/sst2_ptcprl_embedding_qwen3_4b.yaml` — base experiment config