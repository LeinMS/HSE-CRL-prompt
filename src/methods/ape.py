import os
from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from src.models.lm_wrapper import LMWrapper
import openai
from openai import OpenAI


Example = Dict[str, Any]

class PromptResult:
    def __init__(self, template: str, score: float):
        self.template = template
        self.score = score

class APERunner:
    def __init__(
        self,
        model_name: str,
        task: str,
        templates: List[str],
        verbalizer: Dict[str, str],
        device: str,
        num_labels: int,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        text_column: str = "text",
        label_column: str = "label",
        openai_api_key: Optional[str] = None,
        labels: List[str] = [],
        task_for_model: str = "",
    ):
        self.model_name = model_name
        self.task = task
        self.templates = templates
        self.verbalizer = verbalizer
        self.device = device
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.wrapper = LMWrapper(model_name, head_type="masked_lm" if model_name == "roberta-base" else "causal_lm", device=device, verbalizer=verbalizer)
        self.batch_size = batch_size
        self.best_prompt: Optional[PromptResult] = None
        self.text_column = text_column
        self.label_column = label_column
        self.labels = labels
        self.task_for_model = task_for_model
        
        self.openai_client = OpenAI(api_key=openai_api_key)

    def generate_prompts(
        self,
        num_prompts: int = 5,
        model: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 100,
    ) -> List[str]:
        system_prompt_begin = f"""You are a prompt engineering expert. Generate {num_prompts} different prompt templates for the following task:
        Task: {self.task}
        Number of labels: {self.num_labels}
        Label mapping: {self.labels}
        
        Each template should:
        1. Use {{x}} as a placeholder for the input text
        2. Be clear and concise, and short
        3. Help the model understand the task
        4. Be different from other templates
        5. Start with Template:
        Examples:
        """ 
        system_prompt_end =  '\n        '.join(['Template: ' + template for template in self.templates]) +   "\n        Return only the templates, one per line."
        system_prompt = system_prompt_begin + system_prompt_end
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the prompt templates."}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )
            
            generated_text = response.choices[0].message.content
            prompts = [p.strip().replace("Template: ", "") for p in generated_text.split('\n') if p.strip() and p.strip().startswith("Template:")]
            
            self.templates.extend(prompts)
            return prompts
            
        except Exception as e:
            raise Exception(f"Error with OpenAI: {str(e)}")

    @classmethod
    def build(
        cls,
        model_name: str,
        task: str,
        templates: List[str],
        verbalizer: Dict[str, str],
        device: str,
        num_labels: int,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        text_column: str = "text",
        label_column: str = "label",
        openai_api_key: Optional[str] = None,
        labels: List[str] = [],
        task_for_model: str = "",
        **kwargs,
    ):

        return cls(
            model_name=model_name,
            task=task,
            templates=templates,
            verbalizer=verbalizer,
            device=device,
            num_labels=num_labels,
            tokenizer=tokenizer,
            batch_size=batch_size,
            text_column=text_column,
            label_column=label_column,
            openai_api_key=openai_api_key,
            labels=labels,
            task_for_model=task_for_model,
        )

    def search(
        self,
        train_data: List[Example],
        val_data: List[Example],
    ):

        best_acc = -1.0
        best_tpl: Optional[str] = None
        templates = self.generate_prompts(num_prompts=3) + self.templates
        for tpl in tqdm(templates, desc="APE: searching prompts"):
            texts = [tpl.format(x=ex[self.text_column]) + (self.wrapper.tokenizer.mask_token if self.wrapper.head_type == "masked_lm" else "") for ex in val_data]
            scores_list = self.wrapper.score(
                texts,
                batch_size=self.batch_size,
            )
            if isinstance(scores_list, dict):
                scores_list = [scores_list]
            preds = [max(scores, key=scores.get) for scores in scores_list]
            trues = [ex[self.label_column] for ex in val_data]
            correct = sum(p == t for p, t in zip(preds, trues))
            acc = correct / len(trues) if trues else 0.0

            if acc > best_acc:
                best_acc = acc
                best_tpl = tpl

        self.best_prompt = PromptResult(template=best_tpl or "", score=best_acc)

    @property
    def model(self) -> torch.nn.Module:
        return self.wrapper.model
