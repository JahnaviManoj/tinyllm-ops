"""The one entry point for the training DAG (tutorial 3.5).

Run as a script, never imported into a notebook: ZenML derives its source root
from the main module's file, and a Jupyter kernel has none.

  uv run python run_pipeline.py --config configs/exp_100_smoke.yaml --smoke
  uv run python run_pipeline.py --config configs/exp_111.yaml --threshold 0.62
"""

import argparse

from pipelines.training_pipeline import training_pipeline

SMOKE = {"max_steps": 20, "limit": 50, "threshold": 0.0, "strict_behaviors": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument(
        "--threshold", type=float, help="gate_v2 exact-match floor (pre-registered)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 steps, 50 rows, no gates: proves the DAG, never a result",
    )
    args = parser.parse_args()
    if args.smoke:
        training_pipeline(config_path=args.config, **SMOKE)
    elif args.threshold is None:
        parser.error("--threshold is required for a real run (or pass --smoke)")
    else:
        training_pipeline(config_path=args.config, threshold=args.threshold)


if __name__ == "__main__":
    main()
