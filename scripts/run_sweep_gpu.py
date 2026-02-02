#!/usr/bin/env python3
"""
Run a GPU-accelerated parameter sweep from a configuration file.

Usage:
    python scripts/run_sweep_gpu.py configs/killer_final.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_cbd.config import load_config
from vfd_cbd.gpu_sweep import run_sweep_gpu
from vfd_cbd.sweep import get_sweep_summary
from vfd_cbd.io import create_run_folder, save_results

try:
    from vfd_cbd.plots import plot_all

    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False
    print("Warning: Matplotlib not found, plots will be disabled.")


def main():
    parser = argparse.ArgumentParser(description="Run a CBD parameter sweep on GPU.")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output."
    )

    args = parser.parse_args()

    # Load configuration
    if not args.quiet:
        print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir is not None:
        config.run.out_dir = str(args.output_dir)

    # Calculate total points
    total_points = config.sweep.n_df * config.sweep.n_k

    if not args.quiet:
        print(
            f"System: N={config.system.N} oscillators, geometry={config.system.geometry}"
        )
        print(
            f"Sweep: {config.sweep.n_df} x {config.sweep.n_k} = {total_points} points"
        )
        print(f"Simulation: T={config.sim.T}s, dt={config.sim.dt}s")
        print("Running GPU sweep...")

    # Run sweep
    result = run_sweep_gpu(config)

    # Create run folder and save results
    run_path = create_run_folder(config, result.timestamp)
    save_results(result, run_path)

    if not args.quiet:
        print(f"Results saved to: {run_path}")

    # Generate plots
    if not args.no_plots and PLOTS_AVAILABLE:
        if not args.quiet:
            print("Generating plots...")
        plot_paths = plot_all(result, run_path)
        if not args.quiet:
            for name, path in plot_paths.items():
                print(f"  - {name}: {path.name}")
    elif not args.no_plots and not PLOTS_AVAILABLE:
        if not args.quiet:
            print("Skipping plots (matplotlib not available).")

    # Print summary
    summary = get_sweep_summary(result)
    if not args.quiet:
        print()
        print("Summary:")
        print(
            f"  Stable points: {summary['stable_points']}/{summary['total_points']} ({summary['stable_fraction'] * 100:.1f}%)"
        )
        print(f"  Mean coherence: {summary['C_mean_overall']:.3f}")
        print(
            f"  Elapsed time: {summary['elapsed_seconds']:.2f}s ({summary['points_per_second']:.1f} points/s)"
        )

    print(f"\nRun complete: {run_path}")
    return str(run_path)


if __name__ == "__main__":
    main()
