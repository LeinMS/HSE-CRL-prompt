import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import Literal, Dict, Union, List

HeadType = Literal["causal_lm", "masked_lm"]

class LMWrapper:

    def __init__(
        self,
        model_name: str,
        head_type: HeadType = "causal_lm",
        device: str = "cpu",
        verbalizer: Dict[str, List[str]] = None,
        **tokenizer_kwargs,
    ):
        self.model_name = model_name
        self.head_type = head_type
        self.device = device
        self.verbalizer = verbalizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if head_type == "causal_lm":
            self.model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(model_name)
                .to(self.device)
            )
        elif head_type == "masked_lm":
            self.model: PreTrainedModel = (
                AutoModelForMaskedLM.from_pretrained(model_name)
                .to(self.device)
            )
        
    
    def score_verbalizer_causal(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        all_results: List[Dict[str, float]] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs  = self.tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            last_logits = logits[torch.arange(len(batch)), last_token_idx] 
            for i in range(len(batch)):
                scores = {}
                for label, token_arr in self.verbalizer.items():
                    token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in token_arr]
                    scores[label] = last_logits[i, token_ids].mean().cpu().numpy()

                all_results.append(scores)
        return all_results

    def score_verbalizer_masked(
        self,
        masked_texts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        if isinstance(masked_texts, str):
            masked_texts = [masked_texts]
        
        all_results = []
        for i in range(0, len(masked_texts), batch_size):
            batch = masked_texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (inputs.input_ids == mask_token_id).nonzero(as_tuple=False)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

            for b in range(len(batch)):
                pos = mask_positions[mask_positions[:,0] == b][0, 1]
                row = probs[b, pos]
                scores = {}
                for label, token_arr in self.verbalizer.items():
                    token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in token_arr]
                    scores[label] = row[token_ids].sum().item()
                all_results.append(scores)
        return all_results

    def score(self, texts, batch_size):
        if self.head_type == "causal_lm":
            return self.score_verbalizer_causal(texts, batch_size)
        elif self.head_type == "masked_lm":
            return self.score_verbalizer_masked(texts, batch_size)

    def predict_verbalizer_causal(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> Union[str, List[str]]:
        scores = self.score_verbalizer_causal(prompts, batch_size)
        return [max(s, key=s.get) for s in scores]

    def predict_verbalizer_masked(
        self,
        masked_texts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> Union[str, List[str]]:
        scores = self.score_verbalizer_masked(masked_texts, batch_size)
        if isinstance(scores, dict):
            return max(scores, key=scores.get)
        return [max(s, key=s.get) for s in scores]

    def predict(self, texts, batch_size):
        if self.head_type == "causal_lm":
            return self.predict_verbalizer_causal(texts, batch_size)
        elif self.head_type == "masked_lm":
            return self.predict_verbalizer_masked(texts, batch_size)

def main():
    verbalizer = {1: ["Ġpositive"], 0: ["Ġnegative"]}
    wrapper = LMWrapper("tiiuae/falcon-rw-1b", head_type="causal_lm", device="cuda", verbalizer=verbalizer)

    score = wrapper.predict_verbalizer_causal("Review: 'so cool'. Sentiment of review: ", batch_size=1)
    print(score)
    prompts = [
        "Review: bad. Sentiment of review: ",
        "Review: it is terrible. Sentiment of review: "
    ]
    scores_batch = wrapper.predict_verbalizer_causal(prompts, batch_size=2)
    print(wrapper.score_verbalizer_causal(prompts, batch_size=2))
    print(scores_batch)
    wrapper = LMWrapper("roberta-base", head_type="masked_lm", device="cuda", verbalizer=verbalizer)

    score = wrapper.predict_verbalizer_masked("Review: 'so cool'. Sentiment of review (only positive or negative): <mask>.", batch_size=1)
    print(score)
    prompts = [
        "Review: 'it is slow -- very , very slow . '. Sentiment (only positive or negative): <mask>.",
        "Review: 'awful'. Sentiment (only positive or negative): <mask>."
    ]
    scores_batch = wrapper.predict_verbalizer_masked(prompts, batch_size=2)
    print(scores_batch)


if __name__ == "__main__":
    main()