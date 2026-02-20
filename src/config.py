from pathlib import Path

# Resolve the directory of THIS file (src/config.py)
SRC_ROOT = Path(__file__).resolve().parent

# The project root is one level up from src
PROJECT_ROOT = SRC_ROOT.parent


OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

