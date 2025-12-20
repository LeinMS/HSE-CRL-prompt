from typing import List, Optional, Literal, Tuple, Dict, Any
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM
)
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead

TaskStr = Literal["cls", "gen"]


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model

@dataclass
class _RewardConfig:
    temperature: float = 1.0 

class PRewriteRunner:
    def __init__(
        self,
        base_model,
        base_tokenizer,
        rewriter,
        rew_tokenizer,
        task: TaskStr,
        mask_token: str | None,
        text_column: str = "text",
        label_column: str = "label",
    ):
        self.model = base_model  
        self.tokenizer = base_tokenizer
        self.rewriter = rewriter
        self.rew_tok = rew_tokenizer
        self.task = task
        self.mask_token = mask_token

        self.best_prompt: str | None = None
        self._draft_prompt: str = "<PROMPT> <TEXT>"
        self.text_column = text_column
        self.label_column = label_column

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        *,
        draft_prompt: str = "<PROMPT> <TEXT>",
        rewriter_name: str = "google-t5/t5-small",
        rl_steps: int = 2000,
        rl_batch_size: int = 16,
        gen_max_new_tokens: int = 32,
        reward_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        text_column: str = "text",
        label_column: str = "label",
    ):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        base = _freeze(AutoModelForMaskedLM.from_pretrained(model_name)).to(device)

        base_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mask_token = base_tok.mask_token
        if mask_token and mask_token not in base_tok.all_special_tokens:
            base_tok.add_special_tokens({"mask_token": mask_token})
            base.resize_token_embeddings(len(base_tok))

        rew = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(rewriter_name).to(
            device
        )
        rew_tok = AutoTokenizer.from_pretrained(rewriter_name, use_fast=True)

        runner = cls(base, base_tok, rew, rew_tok, task, mask_token, text_column, label_column)
        runner._draft_prompt = draft_prompt
        runner._rl_steps = rl_steps
        runner._rl_bs = rl_batch_size
        runner._gen_max = gen_max_new_tokens
        runner._rw_cfg = _RewardConfig(**(reward_cfg or {}))
        return runner

    def make_prompt(self, text: str) -> str:
        prompt = (self.best_prompt or self._draft_prompt).replace("<TEXT>", text)
        return prompt

    def wrap_collator(self, base_collator):
        def collate(batch):
            for item in batch:
                item[self.text_column] = self.make_prompt(item[self.text_column])
            return base_collator(batch)

        return collate

    def search(self, train_ds, val_ds):
        device = next(self.rewriter.parameters()).device
        cfg = PPOConfig(
            batch_size=self._rl_bs,
            learning_rate=1e-5,
            log_with=None,
            ppo_epochs=4,
            max_grad_norm=1.0,
        )
        ppo = PPOTrainer(config=cfg, model=self.rewriter, tokenizer=self.rew_tok)

        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        def _batch_iter():
            while True:
                for i in range(0, len(indices), self._rl_bs):
                    yield [train_ds[j] for j in indices[i : i + self._rl_bs]]
                random.shuffle(indices)

        _iter = _batch_iter()

        for step in range(self._rl_steps):
            batch = next(_iter)
            texts, labels, targets = self._split_batch(batch)

            rew_inp = self.rew_tok(
                [self._draft_prompt] * len(texts),
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                gen_ids = self.rewriter.generate(
                    **rew_inp,
                    max_new_tokens=self._gen_max,
                    do_sample=True,
                    top_p=0.95,
                )
            prompts = self.rew_tok.batch_decode(gen_ids, skip_special_tokens=True)

            reward = self._compute_reward(prompts, texts, labels, targets).to(device)

            ppo.step(rew_inp["input_ids"], gen_ids, reward)

            if (step + 1) % 100 == 0:
                print(
                    f"[PRewrite] step {step+1}/{self._rl_steps}  reward={reward.mean().item():.4f}"
                )

        self.best_prompt, best_score = self._beam_select(val_ds, beam_size=10)
        print(f"[PRewrite] best prompt: {self.best_prompt!r}  score={best_score:.4f}")

    def _split_batch(self, batch):
        texts = [it.get("sentence", it.get("text")) for it in batch]
        labels = torch.tensor([it.get("label", -1) for it in batch])
        if self.task == "gen":
            if "answer" in batch[0]:
                targets = [it["answer"] for it in batch]
            elif "target" in batch[0]:
                targets = [it["target"] for it in batch]

        else:
            targets = None
        return texts, labels, targets

    def _compute_reward(
        self,
        prompts: List[str],
        texts: List[str],
        labels: torch.Tensor,
        targets: List[str] | None,
    ) -> torch.Tensor:

        full_inputs = [p.replace("<TEXT>", x) for p, x in zip(prompts, texts)]
        enc = self.tokenizer(
            full_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        if self.task == "cls":
            with torch.no_grad():
                logits = self.model(**enc).logits 
            margin = logits.gather(1, labels.unsqueeze(1)).squeeze(1) - logits.masked_fill(
                F.one_hot(labels, logits.size(-1)).bool(), -1e9
            ).max(dim=1).values
            return margin

        with torch.no_grad():
            if self.mask_token:
                mask_pos = (enc["input_ids"] == self.tokenizer.mask_token_id).nonzero()
                logits = self.model(**enc).logits
                sel = logits[mask_pos[:, 0], mask_pos[:, 1]]
                gold = self.tokenizer(targets)["input_ids"]
                gold = torch.tensor([t[0] for t in gold], device=sel.device)
                log_p = F.log_softmax(sel, dim=-1).gather(1, gold.unsqueeze(1)).squeeze(1)
                return log_p
            else:
                with self.tokenizer.as_target_tokenizer():
                    tgt_enc = self.tokenizer(
                        targets,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                out = self.model(**enc, labels=tgt_enc.input_ids)
                neg_nll = -out.loss * tgt_enc.input_ids.size(1)
                ln_len = tgt_enc.input_ids.ne(self.tokenizer.pad_token_id).sum(dim=1)
                score = neg_nll / (ln_len.float() ** self._rw_cfg.temperature)
                return score

    def _beam_select(self, val_ds, beam_size=10) -> Tuple[str, float]:
        device = next(self.rewriter.parameters()).device
        rew_inp = self.rew_tok(
            [self._draft_prompt] * beam_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = self.rewriter.generate(
                **rew_inp,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=self._gen_max,
            )
        cands = self.rew_tok.batch_decode(gen_ids, skip_special_tokens=True)

        best_prompt, best_score = None, -1e9
        texts, labels, targets = self._split_batch(list(val_ds))
        for cand in cands:
            score = (
                self._compute_reward([cand] * len(texts), texts, labels, targets)
                .mean()
                .item()
            )
            if score > best_score:
                best_score, best_prompt = score, cand
        return best_prompt or self._draft_prompt, best_score

    def trainable_param_count(self):
        return sum(p.numel() for p in self.rewriter.parameters() if p.requires_grad)
