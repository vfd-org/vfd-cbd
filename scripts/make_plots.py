#!/usr/bin/env python3
"""
Regenerate plots from an existing run folder.

Usage:
    python scripts/make_plots.py out/run_20240101_120000_abc123/
    python scripts/make_plots.py out/run_20240101_120000_abc123/ --dpi 300
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_cbd.io import load_results
from vfd_cbd.plots import plot_all


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from an existing run folder."
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="Path to run folder."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution (default: 150)."
    )

    args = parser.parse_args()

    if not args.run_path.exists():
        print(f"Error: Run folder not found: {args.run_path}")
        sys.exit(1)

    print(f"Loading results from: {args.run_path}")
    result = load_results(args.run_path)

    print("Generating plots...")
    plot_paths = plot_all(result, args.run_path, dpi=args.dpi)

    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    print("Plots generated.")


if __name__ == "__main__":
    main()
