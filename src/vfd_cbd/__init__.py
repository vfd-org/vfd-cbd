"""
VFD-CBD: Coherence Boundary Diagnostic

A coupled-oscillator network stability and collapse boundary simulator.
This package provides tools for analyzing coherence stability vs parameters
(detuning and coupling-to-loss), producing phase diagrams, collapse boundary
curves, and failure mode classifications.

NOTE: This is strictly a resonant geometry / coupled oscillator network
stability & collapse boundary tool. It makes no claims about propulsion,
gravity, UFOs, or power extraction.
"""

__version__ = "0.1.0"
__author__ = "VFD Research"

from .config import Config, load_config, validate_config
from .geometry import create_geometry, ring_positions, circle_positions
from .coupling import create_coupling_matrix, ring_adjacency, distance_decay
from .dynamics import simulate, rk4_step, compute_derivatives
from .metrics import (
    kuramoto_order_parameter,
    compute_coherence_stats,
    count_phase_slips,
    count_clusters,
    detect_inversion,
)
from .modes import classify_mode, ModeType
from .sweep import run_sweep, SweepResult

try:
    from .plots import plot_phase_diagram, plot_collapse_boundary, plot_mode_map
except ImportError:
    # Handle missing matplotlib
    plot_phase_diagram = None
    plot_collapse_boundary = None
    plot_mode_map = None
from .io import save_results, load_results, create_run_folder, compute_config_hash

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "Config",
    "load_config",
    "validate_config",
    # Geometry
    "create_geometry",
    "ring_positions",
    "circle_positions",
    # Coupling
    "create_coupling_matrix",
    "ring_adjacency",
    "distance_decay",
    # Dynamics
    "simulate",
    "rk4_step",
    "compute_derivatives",
    # Metrics
    "kuramoto_order_parameter",
    "compute_coherence_stats",
    "count_phase_slips",
    "count_clusters",
    "detect_inversion",
    # Modes
    "classify_mode",
    "ModeType",
    # Sweep
    "run_sweep",
    "SweepResult",
    # Plots
    "plot_phase_diagram",
    "plot_collapse_boundary",
    "plot_mode_map",
    # I/O
    "save_results",
    "load_results",
    "create_run_folder",
    "compute_config_hash",
]
