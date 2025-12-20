import pytest
from src.methods.prompt_v1 import PromptV1Runner
import sys 

@pytest.mark.parametrize(
    "model_name, task", [
        ("roberta-base", "cls"),
        ("tiiuae/falcon-rw-1b", "gen"),
    ],
)
def test_prompt_v1_build(model_name, task):
    runner = PromptV1Runner.build(model_name, task, num_virtual_tokens=10)
    param_count = runner.trainable_param_count()
    assert param_count < 5e4