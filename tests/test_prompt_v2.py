import pytest
from src.methods.prompt_v2 import PromptV2Runner

@pytest.mark.parametrize(
    "model_name, task",
    [("roberta-base", "cls"),
     ("tiiuae/falcon-rw-1b", "gen")],
)
def test_prompt_v2_smoke(model_name, task):
    runner = PromptV2Runner.build(model_name, task, num_virtual_tokens=16)
    assert runner.trainable_param_count() > 1e5 
