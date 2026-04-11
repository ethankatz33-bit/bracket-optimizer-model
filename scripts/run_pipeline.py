"""
run_pipeline.py
Orchestrates the full March Madness data pipeline:
  1. load_data.py         → data/raw/ncaa_tournament_games.csv
  2. clean_data.py        → data/processed/cleaned_games.csv
  3. compute_probabilities.py → data/processed/seed_probabilities.json
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
BANNER = "=" * 62


def run(script_name: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{BANNER}")
    print(f"  STEP: {script_name}")
    print(BANNER)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,         # raises CalledProcessError on non-zero exit
    )


def main() -> None:
    print(BANNER)
    print("  March Madness Bracket Optimizer — Data Pipeline")
    print(BANNER)

    steps = [
        "load_data.py",
        "clean_data.py",
        "compute_probabilities.py",
    ]

    for step in steps:
        run(step)

    output = Path(__file__).parent.parent / "data" / "processed" / "seed_probabilities.json"
    print(f"\n{BANNER}")
    print("  Pipeline complete!")
    print(f"  Output: {output}")
    print(BANNER)


if __name__ == "__main__":
    main()
