import logging, pytest, torch
from src.data_utils import get_dataset_bundle, get_data_collator

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "name, tokenizer",
    [
        ("sst2", "roberta-base"),
        ("ag_news", "roberta-base"),
        ("trec", "roberta-base"),
    ],
)
def test_cls_loading(name, tokenizer):
    train, val, nlab, ttype, tok = get_dataset_bundle(name, tokenizer, max_length=64)
    log.info("Loaded %s  train=%d  val=%d  labels=%d", name, len(train), len(val), nlab)

    assert ttype == "cls" and nlab >= 2
    collator = get_data_collator(ttype, tok)
    batch = collator([train[i] for i in range(4)])

    log.debug("input_ids shape %s", batch["input_ids"].shape)
    assert batch["labels"].dtype == torch.long

def test_xsum_loading():
    train, val, _, ttype, tok = get_dataset_bundle("xsum", "t5-small", max_length=32)
    log.info("Loaded XSum  train=%d  val=%d", len(train), len(val))

    assert ttype == "gen"
    collator = get_data_collator(ttype, tok)
    batch = collator([train[i] for i in range(2)])
    assert batch["labels"].shape[1] <= 64