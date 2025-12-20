import importlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import os
import textgrad as tg

TaskStr = Literal["cls", "gen"]


def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class TextGradPromptRunner:

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        *,
        engine_name: str = "gpt-3.5-turbo-0125",
        start_prompt: str = "",
        eval_fn: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "TextGradPromptRunner":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if task == "cls":
            base = _freeze(AutoModelForMaskedLM.from_pretrained(model_name)).to(device)
            mask_tok = base.config.mask_token or "[MASK]"
        elif task == "gen":
            base = _freeze(AutoModelForCausalLM.from_pretrained(model_name)).to(device)
            mask_tok = None     

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if mask_tok and mask_tok not in tok.all_special_tokens:
            tok.add_special_tokens({"mask_token": mask_tok})
            base.resize_token_embeddings(len(tok))


        fn = None
        if eval_fn:
            mod, name = eval_fn.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod), name)

        return cls(base, tok, task, mask_tok, engine_name, start_prompt, fn)

    def __init__(
        self,
        model,
        tokenizer,
        task: TaskStr,
        mask_token: str | None,
        engine_name: str,
        start_prompt: str,
        eval_fn: Optional[Callable],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.mask_token = mask_token
        self.eval_fn = eval_fn

        self.engine = tg.get_engine(engine_name)
        tg.set_backward_engine(self.engine, override=True)

        self.system_prompt = tg.Variable(start_prompt, requires_grad=True, role_description="system prompt")

        self.best_prompt: str | None = None

    def _device(self):
        return next(self.model.parameters()).device

    def _default_reward(self, preds, labels):

        if self.task == "cls":
            return (preds == labels).float()
        return preds.max(-1).values

    def _model_predict(self, batch_texts: List[str]):
        prompts = [f"{self.system_prompt.value} {t}" for t in batch_texts]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self._device())
        with torch.no_grad():
            logits = self.model(**enc).logits
        return logits

    def search(self, train_ds, val_ds, epochs: int = 5):
        from torch.utils.data import DataLoader
        loader = DataLoader(list(train_ds), batch_size=16, shuffle=True)

        best_prompt = self.system_prompt.value
        best_val    = self._quick_val(val_ds)

        for epoch in range(epochs):
            for step, batch in enumerate(loader):
                losses: List[tg.Variable] = []

                for item in zip(batch["input_ids"],
                                batch["labels"]):
                    text, lab = item
                    text = self.tokenizer.decode(text, skip_special_tokens=True)
                    
                    x = tg.Variable(text, requires_grad=False,
                                    role_description="query text")
                    y = tg.Variable(int(lab), requires_grad=False,
                                    role_description="ground-truth label")
                    print(x, y, "xy")
                    prompt   = f"{self.system_prompt.value} {x.value}"
                    logits   = self._model_predict([prompt])[0]

                    pred_idx = int(logits.argmax().item())
                    prediction = (pred_idx)

                    try:
                        out_var = (self.eval_fn or self._default_reward)(
                            pred_idx, y.value
                        )
                        if not isinstance(out_var, tg.Variable):
                            out_var = tg.Variable(out_var,
                                                requires_grad=True,
                                                role_description="scalar reward")
                    except Exception:
                        out_var = self.eval_fn(
                            inputs=dict(prediction=prediction,
                                        ground_truth_answer=y)
                        )
                    losses.append(out_var)

                total_loss = tg.sum(losses)          
                tg.backward(total_loss)

                try:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                except IndexError as e:
                    print(f"[TGD] optimizer step failed: {e}")
                    continue

            val_acc = self._quick_val(val_ds)
            print(f"[TGD] epoch {epoch+1}/{epochs}  val={val_acc:.4f}")

            if val_acc >= best_val:
                best_val = val_acc
                best_prompt = self.prompt.get_value()
            else:
                print("  ↳ revert prompt")
                self.prompt.set_value(best_prompt)

        self.best_prompt = best_prompt
        print("Best prompt:", self.best_prompt)

    def _quick_val(self, val_ds):
        sample = random.sample(list(val_ds), min(100, len(val_ds)))
        texts = [self.tokenizer.decode(b.get("input_ids"), skip_special_tokens=True) for b in sample]
        labels = torch.tensor([s.get("labels", 0) for s in sample], device=self._device())
        logits = self._model_predict(texts)
        pred_idx = int(logits.argmax().item())
        reward = self.eval_fn(pred_idx, labels) if self.eval_fn else self._default_reward(pred_idx, labels)
        return reward.mean().item()

    def make_prompt(self, text: str) -> str:
        return f"{self.best_prompt or self.system_prompt.value} {text}"

    def wrap_collator(self, base_collator):
        def collate(batch):
            for item in batch:
                key = "sentence" if "sentence" in item else "text"
                item[key] = self.make_prompt(item[key])
            return base_collator(batch)
        return collate

    def trainable_param_count(self):
        return len(self.best_prompt.split()) if self.best_prompt else len(self.system_prompt.value.split())

