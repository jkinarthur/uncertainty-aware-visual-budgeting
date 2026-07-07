"""Thin wrapper: evaluate UAViB and baselines. See uavib.cli for flags.

    python scripts/run_eval.py --backend dummy
    python scripts/run_eval.py --backend qwen --device cuda --source real --data-root data/raw
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavib.cli import main_eval  # noqa: E402

if __name__ == "__main__":
    main_eval()
