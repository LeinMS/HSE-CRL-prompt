from typing import Sequence, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    modeling_outputs,
)

TaskStr = Literal["cls"]
ModelKind = Literal["maskedlm", "seqcls"]

def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    labels = labels.unsqueeze(1)                             
    mask = torch.eq(labels, labels.T).float()               
    logits = feats @ feats.T / temp                         
    logits = logits - torch.eye(len(feats), device=feats.device) * 1e4  
    log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
    loss = -(mask * log_prob).sum(1) / mask.sum(1)
    return loss.mean()

class ConsPromptModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_type: ModelKind = "maskedlm",
        num_labels: int | None = None,
        *,
        templates: Optional[Sequence[str]] = None,
        verbalizer: Optional[Sequence[str]] = None,
        soft_prompt_length: int = 20,
        lambda_p: float = 0.5,
        lambda_b: float = 0.5,
        temp: float = 0.07,
        device: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__()

        if model_type == "maskedlm" and (verbalizer is None or len(verbalizer) == 0) or model_type == "seqcls" and (num_labels is None):
            raise ValueError("verbalizer must be provided for 'maskedlm' mode or num_labels must be provided for 'seqcls' mode")

        self.model_type: ModelKind = model_type
        self.lambda_p = lambda_p
        self.lambda_b = lambda_b
        self.temp = temp

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        if model_type == "maskedlm":
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        else:  
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to(device)

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.templates = templates or ["{text}"]
        hidden_size = self.backbone.config.hidden_size
        self.soft_prompt_length = soft_prompt_length
        self.soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, hidden_size, device=device))

        if model_type == "maskedlm":
            self.mask_id = self.tokenizer.mask_token_id
            vocab_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in verbalizer]
            if any(i == self.tokenizer.unk_token_id for i in vocab_ids):
                raise ValueError("All verbalizer tokens must exist in vocab")
            self.register_buffer("verbalizer_ids", torch.tensor(vocab_ids, device=device))
            self.num_labels = len(vocab_ids)
        else:
            self.mask_id = None
            self.verbalizer_ids = None 
            self.num_labels = num_labels 

    def _apply_templates(self, texts: Sequence[str]) -> list[str]:
        return [tpl.format(text=t) for t in texts for tpl in self.templates]

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        texts: Sequence[str] | None = None,
        **kwargs,
    ) -> modeling_outputs.SequenceClassifierOutput:
        
        if texts is None:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        B = len(texts)
        device = self.soft_prompt.device

        prompts = self._apply_templates(texts)  
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation="only_first",
            max_length=512 - self.soft_prompt_length,
            return_tensors="pt",
        ).to(device)
        input_ids_disc = enc.input_ids    
        attn_disc = enc.attention_mask      


        embeds_disc = self.backbone.base_model.embeddings(input_ids_disc)  
        soft = self.soft_prompt.unsqueeze(0).expand(len(prompts), -1, -1) 

        inputs_embeds = torch.cat([soft, embeds_disc], dim=1)  
        attn = torch.cat(
            [torch.ones(len(prompts), self.soft_prompt_length, device=device), attn_disc],
            dim=1,
        )

        outputs = self.backbone(  
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_last = outputs.hidden_states[-1]  

        if self.model_type == "maskedlm":
            mask_positions = (input_ids_disc == self.mask_id).nonzero(as_tuple=True)
            if mask_positions[0].numel() == 0:
                raise ValueError("No [MASK] token found in the discrete templates.")
            idxs = mask_positions[1] + self.soft_prompt_length  
            feats = hidden_last[mask_positions[0], idxs]        
            feats = F.normalize(feats, dim=1)

            mask_logits = outputs.logits[mask_positions[0], idxs]   
            cls_logits = mask_logits[:, self.verbalizer_ids]        
            cls_logits = cls_logits.view(B, len(self.templates), -1).mean(1)  
        else:
            cls_idx = self.soft_prompt_length  
            feats = hidden_last[:, cls_idx, :]                  
            feats = F.normalize(feats, dim=1)

            cls_logits = outputs.logits                         
            cls_logits = cls_logits.view(B, len(self.templates), -1).mean(1)  

        loss = None
        if labels is not None:
            ce = F.cross_entropy(cls_logits, labels)

            prompt_labels = torch.arange(B, device=device).repeat_interleave(len(self.templates))
            scl_p = supcon_loss(feats, prompt_labels, self.temp)

            feats_b = feats.view(B, len(self.templates), -1).mean(1)  
            scl_b = supcon_loss(feats_b, labels, self.temp)

            loss = ce + self.lambda_p * scl_p + self.lambda_b * scl_b

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=cls_logits,
            hidden_states=None,
            attentions=None,
        )


class ConsPromptRunner:

    def __init__(self, model: ConsPromptModel, tokenizer: AutoTokenizer, task: TaskStr):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        num_labels: int,
        *,
        model_type: ModelKind = "maskedlm",
        templates: Sequence[str],
        verbalizer: Optional[Sequence[str]] = None,
        soft_prompt_length: int = 20,
        lambda_p: float = 0.5,
        lambda_b: float = 0.5,
        temp: float = 0.07,
        device: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> "ConsPromptRunner":
        print(num_labels)
        if model_type == "maskedlm":
            if verbalizer is None or len(verbalizer) != num_labels:
                raise ValueError("For 'maskedlm', len must match num_labels")
        else:  
            verbalizer = None

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        model = ConsPromptModel(
            model_name=model_name,
            model_type=model_type,
            num_labels=num_labels,
            templates=templates,
            verbalizer=verbalizer,
            soft_prompt_length=soft_prompt_length,
            lambda_p=lambda_p,
            lambda_b=lambda_b,
            temp=temp,
            device=device,
            tokenizer=tokenizer,
        )
        return cls(model, tokenizer, task)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
