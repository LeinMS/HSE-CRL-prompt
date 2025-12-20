from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

TaskStr = Literal["cls", "gen"]

class SoftPrompt(nn.Module):
    def __init__(self, hidden: int, length: int):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(length, hidden) * 0.02)

    def expand(self, B):
        return self.emb.unsqueeze(0).expand(B, -1, -1)


class _MixSoftCls(nn.Module):
    def __init__(self, base, K: int, T: int):
        super().__init__()
        self.base = base
        H = base.config.hidden_size
        self.prompts = nn.ModuleList([SoftPrompt(H, T) for _ in range(K)])
        self.gate = nn.Linear(H, K, bias=False)
        for p in base.parameters():
            p.requires_grad_(False)

    def _emb_layer(self):
        return getattr(self.base, self.base.base_model_prefix).word_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        B = input_ids.size(0)
        x_emb = self._emb_layer()(input_ids)

        logits_all = []
        cls_vec = None
        for p in self.prompts:
            p_emb = p.expand(B)
            x = torch.cat([p_emb, x_emb], 1)
            attn = torch.cat([
                torch.ones(B, p_emb.size(1), device=attention_mask.device),
                attention_mask,
            ], 1)
            out = self.base(inputs_embeds=x, attention_mask=attn, return_dict=True, output_hidden_states=True)
            logits_all.append(out.logits)
            cls_vec = out.hidden_states[-1][:, 0, :]

        logits_stack = torch.stack(logits_all, 1)
        alpha = torch.softmax(self.gate(cls_vec), -1)
        mixed = (alpha.unsqueeze(-1) * logits_stack).sum(1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(mixed, labels.to(mixed.device))
        return {"loss": loss, "logits": mixed}


class _MixSoftSeq2Seq(nn.Module):
    def __init__(self, base, K: int, T: int):
        super().__init__()
        self.base = base
        H = base.config.d_model
        self.prompts = nn.ModuleList([SoftPrompt(H, T) for _ in range(K)])
        self.gate = nn.Linear(H, K, bias=False)
        for p in base.parameters():
            p.requires_grad_(False)

    def _enc_emb(self):
        return getattr(self.base, self.base.base_model_prefix).word_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        B = input_ids.size(0)
        x_emb = self._enc_emb()(input_ids)

        logits_all = []
        cls_pool = None
        for p in self.prompts:
            p_emb = p.expand(B)
            enc_in = torch.cat([p_emb, x_emb], 1)
            enc_mask = torch.cat([
                torch.ones(B, p_emb.size(1), device=attention_mask.device),
                attention_mask,
            ], 1)
            out = self.base(
                inputs_embeds=enc_in,
                attention_mask=enc_mask,
                labels=labels,
                output_hidden_states=True,
            )
            logits_all.append(out.logits)
            cls_pool = out.encoder_last_hidden_state[:, 0, :]

        logits_stack = torch.stack(logits_all, 1)
        alpha = torch.softmax(self.gate(cls_pool), -1)
        mixed = (alpha.view(B, -1, 1, 1) * logits_stack).sum(1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                mixed.view(-1, mixed.size(-1)),
                labels.to(mixed.device).view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": mixed}

class MixtureSoftRunner:
    def __init__(self, model, tokenizer, task: TaskStr):
        self.model = model
        self.tok = tokenizer
        self.task = task

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        num_templates: int = 2,
        template_len: int = 10,
        device: Optional[str] = None,
        num_labels: int = 2,
        tokenizer=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_name)

        if task == "cls":
            base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            mix = _MixSoftCls(base, num_templates, template_len).to(device)
        elif task == "gen":
            base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            mix = _MixSoftSeq2Seq(base, num_templates, template_len).to(device)

        base.config.pad_token_id = tokenizer.pad_token_id
        return cls(mix, tok, task)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
