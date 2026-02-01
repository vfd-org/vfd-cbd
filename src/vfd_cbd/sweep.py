"""
Parameter sweep functionality for CBD simulations.

Provides functions to run sweeps over detuning and coupling parameters,
collecting coherence metrics and failure modes across the parameter space.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import time

from .config import Config
from .coupling import create_coupling_matrix
from .dynamics import (
    simulate,
    get_phase_offsets,
    get_natural_frequencies,
    get_drive_coupling,
    SimulationResult
)
from .metrics import (
    compute_coherence_stats,
    count_phase_slips_from_coherence,
    count_clusters,
    detect_inversion,
    CoherenceStats
)
from .modes import classify_mode, ModeType


@dataclass
class PointResult:
    """Results for a single parameter point."""
    df_over_f0: float
    k_over_gamma: float
    stats: CoherenceStats
    phase_slip_count: int
    cluster_count: int
    inversion_flag: bool
    mode: ModeType
    stable: bool


@dataclass
class SweepResult:
    """Results from a complete parameter sweep."""
    # Grid parameters
    df_values: NDArray[np.float64]      # Shape (n_df,)
    k_values: NDArray[np.float64]       # Shape (n_k,)

    # Result arrays (all shape (n_k, n_df))
    C_mean: NDArray[np.float64]
    C_min: NDArray[np.float64]
    C_max: NDArray[np.float64]
    C_std: NDArray[np.float64]
    stable_mask: NDArray[np.bool_]
    mode_ids: NDArray[np.int32]
    phase_slip_counts: NDArray[np.int32]
    cluster_counts: NDArray[np.int32]

    # Metadata
    config: Config
    config_hash: str
    timestamp: str
    elapsed_seconds: float
    total_points: int

    def get_point(self, i_k: int, i_df: int) -> PointResult:
        """Get result for a specific grid point."""
        return PointResult(
            df_over_f0=float(self.df_values[i_df]),
            k_over_gamma=float(self.k_values[i_k]),
            stats=CoherenceStats(
                C_mean=float(self.C_mean[i_k, i_df]),
                C_min=float(self.C_min[i_k, i_df]),
                C_max=float(self.C_max[i_k, i_df]),
                C_std=float(self.C_std[i_k, i_df]),
                C_final=float(self.C_mean[i_k, i_df])  # Approximation
            ),
            phase_slip_count=int(self.phase_slip_counts[i_k, i_df]),
            cluster_count=int(self.cluster_counts[i_k, i_df]),
            inversion_flag=False,  # Not stored per-point
            mode=ModeType(self.mode_ids[i_k, i_df]),
            stable=bool(self.stable_mask[i_k, i_df])
        )


def run_sweep(
    config: Config,
    progress_callback: Optional[callable] = None
) -> SweepResult:
    """
    Run a parameter sweep over detuning and coupling.

    Parameters
    ----------
    config : Config
        Complete configuration for the sweep.
    progress_callback : callable, optional
        Called with (current_point, total_points) for progress updates.

    Returns
    -------
    result : SweepResult
        Complete sweep results.
    """
    from .io import compute_config_hash

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract parameters
    N = config.system.N
    geometry = config.system.geometry
    f0_hz = config.drive.f0_hz
    A0 = config.drive.A0
    gamma = config.loss.gamma

    # Simulation parameters
    dt = config.sim.dt
    T = config.sim.T
    sample_every = config.sim.sample_every
    window_frac = config.sim.window_frac

    # Thresholds
    C_threshold = config.thresholds.C_threshold
    slip_threshold = config.thresholds.slip_threshold
    cluster_threshold = config.thresholds.cluster_threshold

    # Sweep grid
    n_df = config.sweep.n_df
    n_k = config.sweep.n_k
    df_values = np.linspace(config.sweep.df_min, config.sweep.df_max, n_df)
    k_values = np.linspace(config.sweep.k_min, config.sweep.k_max, n_k)

    # Drive parameters (base, will be detuned per sweep point)
    D = get_drive_coupling(N, A0)
    phi = get_phase_offsets(N, config.drive.phase_mode, config.drive.phase_custom)

    # Oscillators stay at base frequency (detuning applied to drive instead)
    omega = 2 * np.pi * f0_hz * np.ones(N, dtype=np.float64)

    # Preallocate result arrays
    C_mean_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_min_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_max_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_std_arr = np.zeros((n_k, n_df), dtype=np.float64)
    stable_mask = np.zeros((n_k, n_df), dtype=np.bool_)
    mode_ids = np.zeros((n_k, n_df), dtype=np.int32)
    phase_slip_counts = np.zeros((n_k, n_df), dtype=np.int32)
    cluster_counts = np.zeros((n_k, n_df), dtype=np.int32)

    total_points = n_k * n_df
    current_point = 0

    # Sweep over parameter grid
    for i_k, k_val in enumerate(k_values):
        # Create coupling matrix for this k value
        # k_val represents k/gamma ratio, so actual k = k_val * gamma
        k_strength = k_val * gamma

        K = create_coupling_matrix(
            N=N,
            k_strength=k_strength,
            geometry=geometry,
            neighbor_k=config.system.neighbor_k,
            radius=config.system.radius,
            lambda_decay=config.system.lambda_decay,
            cutoff=config.system.cutoff
        )

        for i_df, df_val in enumerate(df_values):
            # Detune the drive frequency (oscillators stay at base frequency)
            omega_d = 2 * np.pi * f0_hz * (1 + df_val)

            # Run simulation with fixed seed for reproducibility
            # Use config seed + point index for unique but reproducible seed per point
            point_seed = config.run.seed + i_k * n_df + i_df

            result = simulate(
                N=N,
                K=K,
                omega=omega,
                omega_d=omega_d,
                D=D,
                phi=phi,
                dt=dt,
                T=T,
                sample_every=sample_every,
                seed=point_seed
            )

            # Compute metrics
            stats = compute_coherence_stats(result.C_samples, window_frac)
            slip_count = count_phase_slips_from_coherence(result.C_samples)
            clusters = count_clusters(result.theta_final)
            inversion = detect_inversion(
                result.theta_final,
                phi,
                omega_d,
                T
            )

            # Classify mode
            mode = classify_mode(
                stats=stats,
                phase_slip_count=slip_count,
                cluster_count=clusters,
                inversion_flag=inversion,
                C_threshold=C_threshold,
                slip_threshold=slip_threshold,
                cluster_threshold=cluster_threshold
            )

            # Store results
            C_mean_arr[i_k, i_df] = stats.C_mean
            C_min_arr[i_k, i_df] = stats.C_min
            C_max_arr[i_k, i_df] = stats.C_max
            C_std_arr[i_k, i_df] = stats.C_std
            stable_mask[i_k, i_df] = stats.C_mean >= C_threshold
            mode_ids[i_k, i_df] = int(mode)
            phase_slip_counts[i_k, i_df] = slip_count
            cluster_counts[i_k, i_df] = clusters

            current_point += 1
            if progress_callback is not None:
                progress_callback(current_point, total_points)

    elapsed = time.time() - start_time
    config_hash = compute_config_hash(config)

    return SweepResult(
        df_values=df_values,
        k_values=k_values,
        C_mean=C_mean_arr,
        C_min=C_min_arr,
        C_max=C_max_arr,
        C_std=C_std_arr,
        stable_mask=stable_mask,
        mode_ids=mode_ids,
        phase_slip_counts=phase_slip_counts,
        cluster_counts=cluster_counts,
        config=config,
        config_hash=config_hash,
        timestamp=timestamp,
        elapsed_seconds=elapsed,
        total_points=total_points
    )


def find_collapse_boundary(
    result: SweepResult,
    C_threshold: Optional[float] = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find the collapse boundary curve.

    For each detuning value, find the minimum k where coherence
    crosses the threshold.

    Parameters
    ----------
    result : SweepResult
        Sweep results.
    C_threshold : float, optional
        Threshold for collapse detection. If None, uses config value.

    Returns
    -------
    df_boundary : ndarray
        Detuning values at boundary points.
    k_boundary : ndarray
        Coupling values at boundary points.
    """
    if C_threshold is None:
        C_threshold = result.config.thresholds.C_threshold

    df_boundary = []
    k_boundary = []

    for i_df, df_val in enumerate(result.df_values):
        # Find first k where C_mean >= threshold (from low to high)
        C_column = result.C_mean[:, i_df]

        # Look for transition from below to above threshold
        above_threshold = C_column >= C_threshold

        if np.any(above_threshold):
            # Find first index above threshold
            first_above = np.argmax(above_threshold)
            if first_above > 0:
                # Interpolate between last below and first above
                k_low = result.k_values[first_above - 1]
                k_high = result.k_values[first_above]
                C_low = C_column[first_above - 1]
                C_high = C_column[first_above]

                # Linear interpolation
                if C_high != C_low:
                    alpha = (C_threshold - C_low) / (C_high - C_low)
                    k_boundary_val = k_low + alpha * (k_high - k_low)
                else:
                    k_boundary_val = k_low

                df_boundary.append(df_val)
                k_boundary.append(k_boundary_val)
            elif first_above == 0:
                # Already above threshold at lowest k
                df_boundary.append(df_val)
                k_boundary.append(result.k_values[0])
        else:
            # Never reaches threshold - system always unstable for this df
            # Don't add a point
            pass

    return np.array(df_boundary), np.array(k_boundary)


def get_sweep_summary(result: SweepResult) -> dict:
    """
    Get a summary of sweep results.

    Parameters
    ----------
    result : SweepResult
        Sweep results.

    Returns
    -------
    summary : dict
        Summary statistics.
    """
    from .modes import ModeType, mode_to_string

    # Count modes
    mode_counts = {}
    for mode in ModeType:
        count = np.sum(result.mode_ids == int(mode))
        if count > 0:
            mode_counts[mode_to_string(mode)] = int(count)

    return {
        "total_points": result.total_points,
        "stable_points": int(np.sum(result.stable_mask)),
        "unstable_points": int(np.sum(~result.stable_mask)),
        "stable_fraction": float(np.mean(result.stable_mask)),
        "C_mean_overall": float(np.mean(result.C_mean)),
        "C_mean_stable_region": float(np.mean(result.C_mean[result.stable_mask])) if np.any(result.stable_mask) else 0.0,
        "mode_counts": mode_counts,
        "elapsed_seconds": result.elapsed_seconds,
        "points_per_second": result.total_points / result.elapsed_seconds if result.elapsed_seconds > 0 else 0.0
    }
