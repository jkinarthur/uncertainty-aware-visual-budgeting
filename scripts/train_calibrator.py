"""Thin wrapper: train the UAViB calibrator. See uavib.cli for flags.

    python scripts/train_calibrator.py --backend dummy
    python scripts/train_calibrator.py --backend qwen --device cuda --source real --data-root data/raw
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavib.cli import main_train_calibrator  # noqa: E402

if __name__ == "__main__":
    main_train_calibrator()
