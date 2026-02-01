"""
Plotting functions for CBD simulation results.

Provides functions to generate phase diagrams, collapse boundary curves,
and failure mode maps.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from .sweep import SweepResult, find_collapse_boundary
from .modes import ModeType, MODE_NAMES, MODE_COLORS


def plot_phase_diagram(
    result: SweepResult,
    save_path: Optional[Union[str, Path]] = None,
    show_contour: bool = True,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Plot the phase diagram heatmap of coherence.

    Parameters
    ----------
    result : SweepResult
        Sweep results to plot.
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    show_contour : bool, optional
        Whether to overlay the collapse boundary contour (default True).
    figsize : tuple, optional
        Figure size in inches (default (10, 8)).
    dpi : int, optional
        Resolution for saving (default 150).
    cmap : str, optional
        Colormap name (default "viridis").

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid for pcolormesh
    df_grid, k_grid = np.meshgrid(result.df_values, result.k_values)

    # Plot heatmap
    im = ax.pcolormesh(
        df_grid, k_grid, result.C_mean,
        cmap=cmap, vmin=0, vmax=1, shading='auto'
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Mean Coherence $C$')

    # Overlay contour at threshold
    if show_contour:
        C_threshold = result.config.thresholds.C_threshold
        contour = ax.contour(
            df_grid, k_grid, result.C_mean,
            levels=[C_threshold],
            colors=['white'],
            linewidths=2,
            linestyles='--'
        )
        ax.clabel(contour, fmt=f'$C^*={C_threshold}$', fontsize=10)

    # Labels and title
    ax.set_xlabel(r'Detuning $\Delta f / f_0$', fontsize=12)
    ax.set_ylabel(r'Coupling-to-Loss $k / \gamma$', fontsize=12)
    ax.set_title('Coherence Phase Diagram', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_collapse_boundary(
    result: SweepResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot the collapse boundary curve.

    Parameters
    ----------
    result : SweepResult
        Sweep results to plot.
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, optional
        Figure size in inches (default (10, 8)).
    dpi : int, optional
        Resolution for saving (default 150).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Find collapse boundary
    df_boundary, k_boundary = find_collapse_boundary(result)

    C_threshold = result.config.thresholds.C_threshold

    # Plot boundary curve
    if len(df_boundary) > 0:
        ax.plot(
            df_boundary, k_boundary,
            'b-', linewidth=2.5, label=f'Collapse Boundary ($C^*={C_threshold}$)'
        )
        ax.fill_between(
            df_boundary, k_boundary, result.k_values.max(),
            alpha=0.3, color='green', label='Stable Region'
        )
        ax.fill_between(
            df_boundary, result.k_values.min(), k_boundary,
            alpha=0.3, color='red', label='Unstable Region'
        )
    else:
        ax.text(
            0.5, 0.5, 'No collapse boundary found\n(system always stable or unstable)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12
        )

    # Labels and title
    ax.set_xlabel(r'Detuning $\Delta f / f_0$', fontsize=12)
    ax.set_ylabel(r'Coupling-to-Loss $k / \gamma$', fontsize=12)
    ax.set_title('Coherence Collapse Boundary', fontsize=14)

    # Set axis limits
    ax.set_xlim(result.df_values.min(), result.df_values.max())
    ax.set_ylim(result.k_values.min(), result.k_values.max())

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_mode_map(
    result: SweepResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot the failure mode map.

    Parameters
    ----------
    result : SweepResult
        Sweep results to plot.
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, optional
        Figure size in inches (default (10, 8)).
    dpi : int, optional
        Resolution for saving (default 150).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap for modes
    mode_list = [
        ModeType.STABLE,
        ModeType.DECOHERENT,
        ModeType.PHASE_SLIP,
        ModeType.CLUSTER_SPLIT,
        ModeType.INVERSION,
        ModeType.UNKNOWN
    ]
    colors = [MODE_COLORS[m] for m in mode_list]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Create meshgrid for pcolormesh
    df_grid, k_grid = np.meshgrid(result.df_values, result.k_values)

    # Plot mode map
    im = ax.pcolormesh(
        df_grid, k_grid, result.mode_ids,
        cmap=cmap, norm=norm, shading='auto'
    )

    # Create legend patches
    patches = []
    mode_ids_present = np.unique(result.mode_ids)
    for mode in mode_list:
        if int(mode) in mode_ids_present:
            patches.append(
                mpatches.Patch(
                    color=MODE_COLORS[mode],
                    label=MODE_NAMES[mode].replace('_', ' ').title()
                )
            )

    ax.legend(handles=patches, loc='upper right', fontsize=10)

    # Labels and title
    ax.set_xlabel(r'Detuning $\Delta f / f_0$', fontsize=12)
    ax.set_ylabel(r'Coupling-to-Loss $k / \gamma$', fontsize=12)
    ax.set_title('Failure Mode Map', fontsize=14)

    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_all(
    result: SweepResult,
    output_dir: Union[str, Path],
    dpi: int = 150
) -> dict[str, Path]:
    """
    Generate all standard plots and save to output directory.

    Parameters
    ----------
    result : SweepResult
        Sweep results to plot.
    output_dir : str or Path
        Directory to save plots.
    dpi : int, optional
        Resolution for saving (default 150).

    Returns
    -------
    paths : dict
        Dictionary mapping plot names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Phase diagram
    phase_path = output_dir / "plot_phase_diagram.png"
    fig = plot_phase_diagram(result, save_path=phase_path, dpi=dpi)
    plt.close(fig)
    paths["phase_diagram"] = phase_path

    # Collapse boundary
    boundary_path = output_dir / "plot_collapse_boundary.png"
    fig = plot_collapse_boundary(result, save_path=boundary_path, dpi=dpi)
    plt.close(fig)
    paths["collapse_boundary"] = boundary_path

    # Mode map
    mode_path = output_dir / "plot_mode_map.png"
    fig = plot_mode_map(result, save_path=mode_path, dpi=dpi)
    plt.close(fig)
    paths["mode_map"] = mode_path

    return paths


def plot_coherence_timeseries(
    C_samples: NDArray[np.float64],
    t_samples: NDArray[np.float64],
    C_threshold: float = 0.7,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot coherence time series for a single simulation.

    Parameters
    ----------
    C_samples : ndarray
        Coherence values over time.
    t_samples : ndarray
        Time points.
    C_threshold : float, optional
        Stability threshold to show (default 0.7).
    save_path : str or Path, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size in inches.
    dpi : int, optional
        Resolution for saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t_samples, C_samples, 'b-', linewidth=1.5, label='Coherence $C(t)$')
    ax.axhline(C_threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold $C^*={C_threshold}$')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Coherence $C$', fontsize=12)
    ax.set_title('Coherence Time Series', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
