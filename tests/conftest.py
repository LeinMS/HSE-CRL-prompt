import sys
from pathlib import Path
import logging
import pytest

@pytest.fixture(scope="session", autouse=True)
def _file_logging(tmp_path_factory):
    log_file = Path(tmp_path_factory.getbasetemp()) / "tests.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    print(f"[pytest] logs → {log_file}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
