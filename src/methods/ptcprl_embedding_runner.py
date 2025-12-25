from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader
from transformers import AutoModel, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

TaskStr = Literal["cls"]


@dataclass
class PtcprlConfig:
    num_virtual_tokens: int = 20
    phase1_epochs: int = 3
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.05
    rl_subset_size: int = 32
    k_negatives: int = 1
    sigma: float = 2e-5


class SoftPromptEmbeddingClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int, prompt_length: int):
        super().__init__()
        self.base = base_model
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, base_model.config.hidden_size) * 0.02
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        self.config = base_model.config
        self.config.num_labels = num_labels

    def _prepend_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        token_embeds = self.base.get_input_embeddings()(input_ids)
        batch_size = token_embeds.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, token_embeds], dim=1)
        prompt_mask = torch.ones(
            (batch_size, prompt.size(1)),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        inputs_embeds, attention_mask = self._prepend_prompt(input_ids, attention_mask)
        outputs = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class TwoPhaseTrainer(Trainer):
    def __init__(
        self,
        *args,
        phase1_epochs: int = 3,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.05,
        rl_subset_size: int = 32,
        rl_dataset=None,
        k_negatives: int = 1,
        sigma: float = 2e-5,
        **kwargs,
    ):
        self.phase1_epochs = phase1_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rl_subset_size = rl_subset_size
        self.rl_dataset = rl_dataset
        self.k_negatives = k_negatives
        self.sigma = sigma
        super().__init__(*args, **kwargs)

        subset = list(range(min(len(self.rl_dataset), self.rl_subset_size)))
        self._rl_loader = DataLoader(
            self.rl_dataset.select(subset),
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
        )
        self.prompt_params = []
        for name, param in self.model.named_parameters():
            if "prompt_embeddings" in name:
                self.prompt_params.append((name, param))

    def mutate_prompt_for_contrastive(self):
        negatives = []
        for _ in range(self.k_negatives):
            mutated = {}
            for name, param in self.prompt_params:
                mask = (torch.rand_like(param) > 0.1).float()
                mutated[name] = param * mask
            negatives.append(mutated)
        return negatives

    def compute_contrastive_loss(self, logits_pos, logits_negs, temp=1.0):
        pos_norm = F.normalize(logits_pos, dim=-1)
        neg_norms = [F.normalize(n, dim=-1) for n in logits_negs]
        sim_pos = (pos_norm * pos_norm).sum(-1) / temp
        sim_negs = torch.stack([(pos_norm * neg).sum(-1) for neg in neg_norms], dim=1) / temp
        loss = -torch.log(sim_pos.exp() / (sim_pos.exp() + sim_negs.exp().sum(1))).mean()
        return loss

    @torch.no_grad()
    def compute_reward(self, model):
        was_train = model.training
        model.eval()
        correct = total = 0
        for batch in self._rl_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch).logits.argmax(-1)
            correct += (out == batch["labels"]).sum().item()
            total += out.size(0)
        if was_train:
            model.train()
        return correct / max(total, 1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss_mle = outputs.loss

        if (self.state.epoch < self.phase1_epochs) or not model.training:
            return (loss_mle, outputs) if return_outputs else loss_mle

        logits_pos = outputs.logits
        original = {n: p.data.clone() for n, p in self.prompt_params}

        logits_negs = []
        for neg in self.mutate_prompt_for_contrastive():
            with torch.no_grad():
                for name, param in self.prompt_params:
                    param.data.copy_(neg[name].data)
            logits_negs.append(model(**inputs).logits.detach())

        with torch.no_grad():
            for name, param in self.prompt_params:
                param.data.copy_(original[name])

        loss_contrast = self.compute_contrastive_loss(logits_pos, logits_negs)

        if self.state.global_step % 100 == 0:
            log_probs = []
            for name, param in self.prompt_params:
                eps = torch.randn_like(param) * self.sigma
                noisy = param + eps
                dist = Normal(loc=param, scale=self.sigma)
                log_probs.append(dist.log_prob(noisy).sum())
                with torch.no_grad():
                    param.data.copy_(noisy.data)

            reward = self.compute_reward(model)

            with torch.no_grad():
                for name, param in self.prompt_params:
                    param.data.copy_(original[name])

            total_log_prob = torch.stack(log_probs).mean()
            loss_rl = -reward * total_log_prob
        else:
            loss_rl = 0.0

        loss = self.alpha * loss_mle + self.beta * loss_contrast + self.gamma * loss_rl
        return (loss, outputs) if return_outputs else loss


class PTCPRLEmbeddingRunner:
    def __init__(
        self,
        model: SoftPromptEmbeddingClassifier,
        task: TaskStr,
        cfg: PtcprlConfig,
    ):
        self.model = model
        self.task = task
        self.cfg = cfg

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        *,
        num_virtual_tokens: int = 20,
        phase1_epochs: int = 3,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.05,
        rl_subset_size: int = 32,
        k_negatives: int = 1,
        sigma: float = 2e-5,
        device: Optional[str] = None,
        num_labels: int = 2,
        tokenizer=None,
    ) -> "PTCPRLEmbeddingRunner":
        if task != "cls":
            raise ValueError("PTCPRL embedding runner supports classification tasks only.")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModel.from_pretrained(model_name).to(device)
        for param in base.parameters():
            param.requires_grad_(False)
        model = SoftPromptEmbeddingClassifier(
            base_model=base,
            num_labels=num_labels,
            prompt_length=num_virtual_tokens,
        ).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id if tokenizer else None
        cfg = PtcprlConfig(
            num_virtual_tokens=num_virtual_tokens,
            phase1_epochs=phase1_epochs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rl_subset_size=rl_subset_size,
            k_negatives=k_negatives,
            sigma=sigma,
        )
        return cls(model, task, cfg)

    def build_trainer(self, train_ds, val_ds, collator, args, compute_metrics):
        rl_subset = val_ds.select(range(min(len(val_ds), self.cfg.rl_subset_size)))
        return TwoPhaseTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            phase1_epochs=self.cfg.phase1_epochs,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            rl_subset_size=self.cfg.rl_subset_size,
            rl_dataset=rl_subset,
            k_negatives=self.cfg.k_negatives,
            sigma=self.cfg.sigma,
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
