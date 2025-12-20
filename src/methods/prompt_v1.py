from typing import Optional, Literal

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from peft import PromptTuningConfig, TaskType, get_peft_model

TaskStr = Literal["cls", "gen"]


class PromptV1Runner:
    def __init__(self, model, task: TaskStr):
        self.model = model
        self.task = task

    @classmethod
    def build(
        cls,
        model_name: str,
        task: TaskStr,
        num_virtual_tokens: int = 20,
        init_text: Optional[str] = None,
        device: Optional[str] = None,
        num_labels: int = 2,
        tokenizer=None
    ) :
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if task == "cls":
            base = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            peft_task = TaskType.SEQ_CLS
        elif task == "gen":
            try:
                base = AutoModelForCausalLM.from_pretrained(model_name)
                peft_task = TaskType.CAUSAL_LM
            except ValueError:
                base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                peft_task = TaskType.SEQ_2_SEQ_LM

        for p in base.parameters():           
            p.requires_grad = False

        cfg = PromptTuningConfig(
            task_type=peft_task,
            num_virtual_tokens=num_virtual_tokens,
        )
        model = get_peft_model(base, cfg)
        if task == "cls":
            for n, p in base.named_parameters(): 
                if n.startswith('classifier.'):
                    p.requires_grad_(True)
        model.config.pad_token_id = tokenizer.pad_token_id
        return cls(model, task)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)