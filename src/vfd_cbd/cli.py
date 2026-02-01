"""
Command-line interface for CBD simulations.

Provides commands for running sweeps, generating plots, and exporting results.
"""

from pathlib import Path
from typing import Optional
import sys

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

app = typer.Typer(
    name="vfd-cbd",
    help="Coherence Boundary Diagnostic: A coupled-oscillator network stability simulator.",
    add_completion=False
)
console = Console()


@app.command()
def sweep(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir", "-o",
        help="Override output directory from config."
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip plot generation."
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output."
    )
):
    """
    Run a parameter sweep from a configuration file.

    Produces coherence phase diagrams, collapse boundary curves, and
    failure mode maps.
    """
    from .config import load_config
    from .sweep import run_sweep, get_sweep_summary
    from .io import create_run_folder, save_results
    from .plots import plot_all

    # Load configuration
    if not quiet:
        console.print(f"[bold blue]Loading config:[/] {config}")

    cfg = load_config(config)

    # Override output directory if specified
    if out_dir is not None:
        cfg.run.out_dir = str(out_dir)

    # Calculate total points
    total_points = cfg.sweep.n_df * cfg.sweep.n_k

    if not quiet:
        console.print(f"[bold]System:[/] N={cfg.system.N} oscillators, geometry={cfg.system.geometry}")
        console.print(f"[bold]Sweep:[/] {cfg.sweep.n_df} x {cfg.sweep.n_k} = {total_points} points")
        console.print(f"[bold]Simulation:[/] T={cfg.sim.T}s, dt={cfg.sim.dt}s")

    # Run sweep with progress bar
    if not quiet:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running sweep...", total=total_points)

            def update_progress(current, total):
                progress.update(task, completed=current)

            result = run_sweep(cfg, progress_callback=update_progress)
    else:
        result = run_sweep(cfg)

    # Create run folder and save results
    run_path = create_run_folder(cfg, result.timestamp)
    save_results(result, run_path)

    if not quiet:
        console.print(f"[bold green]Results saved to:[/] {run_path}")

    # Generate plots
    if not no_plots:
        if not quiet:
            console.print("[bold blue]Generating plots...[/]")
        plot_paths = plot_all(result, run_path)
        if not quiet:
            for name, path in plot_paths.items():
                console.print(f"  - {name}: {path.name}")

    # Print summary
    if not quiet:
        summary = get_sweep_summary(result)
        console.print()
        console.print("[bold]Summary:[/]")
        console.print(f"  Stable points: {summary['stable_points']}/{summary['total_points']} ({summary['stable_fraction']*100:.1f}%)")
        console.print(f"  Mean coherence: {summary['C_mean_overall']:.3f}")
        console.print(f"  Elapsed time: {summary['elapsed_seconds']:.2f}s ({summary['points_per_second']:.1f} points/s)")

        if summary['mode_counts']:
            console.print("  Mode counts:")
            for mode, count in summary['mode_counts'].items():
                console.print(f"    - {mode}: {count}")

    console.print(f"\n[bold green]Run complete:[/] {run_path}")


@app.command()
def plot(
    run: Path = typer.Option(
        ...,
        "--run", "-r",
        help="Path to run folder.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    dpi: int = typer.Option(
        150,
        "--dpi",
        help="Plot resolution."
    )
):
    """
    Regenerate plots from an existing run folder.
    """
    from .io import load_results
    from .plots import plot_all

    console.print(f"[bold blue]Loading results from:[/] {run}")
    result = load_results(run)

    console.print("[bold blue]Generating plots...[/]")
    plot_paths = plot_all(result, run, dpi=dpi)

    for name, path in plot_paths.items():
        console.print(f"  - {name}: {path}")

    console.print("[bold green]Plots generated.[/]")


@app.command()
def export(
    run: Path = typer.Option(
        ...,
        "--run", "-r",
        help="Path to run folder.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    format: str = typer.Option(
        "json,csv",
        "--format", "-f",
        help="Comma-separated list of formats to export (json, csv)."
    )
):
    """
    Export results from a run folder in specified formats.
    """
    from .io import export_results

    formats = [f.strip() for f in format.split(",")]

    console.print(f"[bold blue]Exporting from:[/] {run}")
    console.print(f"[bold]Formats:[/] {', '.join(formats)}")

    paths = export_results(run, formats)

    for fmt, path in paths.items():
        console.print(f"  - {fmt}: {path}")

    console.print("[bold green]Export complete.[/]")


@app.command()
def info(
    run: Path = typer.Option(
        ...,
        "--run", "-r",
        help="Path to run folder.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    )
):
    """
    Display information about a run.
    """
    from .io import load_results
    from .sweep import get_sweep_summary

    result = load_results(run)
    summary = get_sweep_summary(result)

    table = Table(title=f"Run: {run.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Config Hash", result.config_hash)
    table.add_row("Timestamp", result.timestamp)
    table.add_row("N (oscillators)", str(result.config.system.N))
    table.add_row("Geometry", result.config.system.geometry)
    table.add_row("Grid Size", f"{result.config.sweep.n_df} x {result.config.sweep.n_k}")
    table.add_row("Total Points", str(summary['total_points']))
    table.add_row("Stable Points", f"{summary['stable_points']} ({summary['stable_fraction']*100:.1f}%)")
    table.add_row("Mean Coherence", f"{summary['C_mean_overall']:.4f}")
    table.add_row("Elapsed Time", f"{summary['elapsed_seconds']:.2f}s")

    console.print(table)

    if summary['mode_counts']:
        mode_table = Table(title="Mode Distribution")
        mode_table.add_column("Mode", style="cyan")
        mode_table.add_column("Count", style="green")
        for mode, count in summary['mode_counts'].items():
            mode_table.add_row(mode, str(count))
        console.print(mode_table)


@app.command()
def list_runs(
    out_dir: Path = typer.Option(
        Path("out"),
        "--out-dir", "-o",
        help="Output directory to search."
    )
):
    """
    List all runs in an output directory.
    """
    from .io import list_runs as _list_runs

    runs = _list_runs(out_dir)

    if not runs:
        console.print(f"[yellow]No runs found in {out_dir}[/]")
        return

    table = Table(title=f"Runs in {out_dir}")
    table.add_column("#", style="dim")
    table.add_column("Run Name", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Hash", style="yellow")

    for i, run_path in enumerate(runs, 1):
        # Parse run name: run_YYYYmmdd_HHMMSS_hash
        parts = run_path.name.split("_")
        if len(parts) >= 4:
            timestamp = f"{parts[1]}_{parts[2]}"
            hash_val = parts[3] if len(parts) > 3 else ""
        else:
            timestamp = ""
            hash_val = ""
        table.add_row(str(i), run_path.name, timestamp, hash_val)

    console.print(table)


@app.command()
def version():
    """
    Display version information.
    """
    from . import __version__
    console.print(f"vfd-cbd version {__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
