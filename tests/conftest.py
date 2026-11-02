import sys
from pathlib import Path

# Modules in src/ import each other as top-level modules
# (e.g. ``from features import ...``), so put src/ on the path.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
