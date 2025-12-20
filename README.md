# CRL-Prompt: Contrastive and Reinforcement Learning for Soft Prompt Tuning of Language Models

**CRL-Prompt** is a universal tool for automated prompt generation, optimization, and evaluation for large language models (LLMs). This repository contains the implementation for the paper *CRL-Prompt: Contrastive and Reinforcement Learning for Soft Prompt Tuning of Language Models*, submitted to ARR.

## About the Project

This project implements a framework for automatically developing prompts aimed at solving text classification and generation tasks. Main features include:

* **Prompt Optimization Methods**:

  * PromptV1 and PromptV2 (hard token prompts).
  * Mixture Soft Prompting (blended soft tokens).
  * TextGrad (gradient-based optimization).
  * Consprompt (contrastive optimization).
  * Prewrite RL (reinforcement learning).
  * APE (automated evolutionary search).
* **Experiment Support** via Hydra + YAML configurations.
* **Integration** with `transformers`, `datasets`, `evaluate`, and `openai`.
* **Unit Tests** based on `pytest` for each method implementation.

## Examples of Using a Custom Method

The implementation of our custom method for Falcon and RoBERTa-base on AG News, SST-2, and TREC can be found in the notebooks:

* `src/methods/ptcprl_method_roberta.ipynb`
* `src/methods/ptcprl_method_falcon.ipynb`

## Dependencies

```bash
> pytest
> torch
> transformers
> datasets
> evaluate
> hydra-core
> omegaconf
> tqdm
> openai
```

## Project Structure

```
.
├── configs/               # YAML configuration files for experiments
├── src/                   # Source code of the framework
│   ├── run_experiment.py  # Hydra entry point
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── callbacks/         # Metric tracking callbacks per epoch
│   └── methods/           # Implementations of prompt optimization methods
├── utils/                 # Helper functions
├── tests/                 # Tests (pytest)
├── run_all_experiments.ps1# Script to run all configurations (Windows)
├── pytest.ini             # Pytest configuration
└── README.md              # This file
```

## Usage

### Configuration

All `.yaml` files in the `configs` folder describe the combination:

```
<dataset>_<method>_<model>.yaml
```

For example: `ag_news_prompt_v1_roberta_base.yaml`.

### Running an Experiment

```bash
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base
```

To partially override parameters:

```bash
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base \
    method_cfg.num_candidates=10 training.max_epochs=3
```

### Results

* Training logs are saved to `outputs/exp_name/YYYY-MM-DD_HH-MM-SS/`.
* Epoch metrics are stored in `epoch_metrics.jsonl`.

## Testing

```bash
pytest
```

## Adding a New Method

1. Create a new Runner class in `src/methods`, following the pattern of existing ones.
2. Add an entry to the `_METHODS` dictionary in `src/run_experiment.py`.
3. Write a YAML config file in `configs`.
4. Add tests under `tests/`.
