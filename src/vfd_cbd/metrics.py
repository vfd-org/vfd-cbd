"""
Coherence metrics for coupled oscillator analysis.

Provides functions to compute the Kuramoto order parameter and
related statistics for stability assessment.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass


@dataclass
class CoherenceStats:
    """Statistics computed from coherence time series."""

    C_mean: float  # Mean coherence over analysis window
    C_min: float  # Minimum coherence over analysis window
    C_max: float  # Maximum coherence over analysis window
    C_std: float  # Standard deviation over analysis window
    C_final: float  # Final coherence value


def kuramoto_order_parameter(theta: NDArray[np.float64]) -> float:
    """
    Compute the Kuramoto order parameter (coherence).

    C = |(1/N) Σ exp(i θ_j)|

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Phases of all oscillators.

    Returns
    -------
    C : float
        Coherence value in [0, 1].
        C = 1 means perfect synchronization.
        C = 0 means completely desynchronized.
    """
    N = len(theta)
    if N == 0:
        return 0.0
    complex_phases = np.exp(1j * theta)
    C = np.abs(np.mean(complex_phases))
    return float(C)


def kuramoto_complex_order(theta: NDArray[np.float64]) -> complex:
    """
    Compute the complex Kuramoto order parameter.

    Z = (1/N) Σ exp(i θ_j)

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Phases of all oscillators.

    Returns
    -------
    Z : complex
        Complex order parameter.
        |Z| is the coherence, arg(Z) is the mean phase.
    """
    N = len(theta)
    if N == 0:
        return 0.0 + 0.0j
    complex_phases = np.exp(1j * theta)
    Z = np.mean(complex_phases)
    return Z


def compute_coherence_stats(
    C_samples: NDArray[np.float64], window_frac: float = 0.2
) -> CoherenceStats:
    """
    Compute coherence statistics over the analysis window.

    Parameters
    ----------
    C_samples : ndarray of shape (n_samples,)
        Coherence values sampled over time.
    window_frac : float, optional
        Fraction of time to use for analysis window (default 0.2).
        Uses the last window_frac of samples.

    Returns
    -------
    stats : CoherenceStats
        Computed statistics.
    """
    n_samples = len(C_samples)
    if n_samples == 0:
        return CoherenceStats(C_mean=0.0, C_min=0.0, C_max=0.0, C_std=0.0, C_final=0.0)

    # Get analysis window (last window_frac of samples)
    window_start = int(n_samples * (1 - window_frac))
    window_start = max(0, window_start)
    window = C_samples[window_start:]

    if len(window) == 0:
        window = C_samples[-1:]

    return CoherenceStats(
        C_mean=float(np.mean(window)),
        C_min=float(np.min(window)),
        C_max=float(np.max(window)),
        C_std=float(np.std(window)),
        C_final=float(C_samples[-1]),
    )


def count_phase_slips(
    theta_history: NDArray[np.float64], dt: float, threshold: float = np.pi
) -> int:
    """
    Count phase slip events.

    A phase slip is detected when the wrapped phase difference between
    consecutive time steps exceeds a threshold.

    Parameters
    ----------
    theta_history : ndarray of shape (n_samples, N)
        Phase history over time for all oscillators.
    dt : float
        Time step between samples.
    threshold : float, optional
        Phase change threshold for slip detection (default π).

    Returns
    -------
    count : int
        Total number of phase slip events across all oscillators.
    """
    if theta_history.shape[0] < 2:
        return 0

    # Compute phase differences between consecutive samples
    dtheta = np.diff(theta_history, axis=0)

    # Wrap to [-π, π]
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi

    # Count slips where |dtheta| > threshold
    slips = np.sum(np.abs(dtheta) > threshold)

    return int(slips)


def count_phase_slips_from_coherence(
    C_samples: NDArray[np.float64], drop_threshold: float = 0.3
) -> int:
    """
    Estimate phase slip count from coherence drops.

    A simpler metric that counts significant drops in coherence,
    which may indicate phase slip events.

    Parameters
    ----------
    C_samples : ndarray of shape (n_samples,)
        Coherence values over time.
    drop_threshold : float, optional
        Coherence drop threshold for slip detection (default 0.3).

    Returns
    -------
    count : int
        Estimated number of phase slip events.
    """
    if len(C_samples) < 2:
        return 0

    # Compute coherence drops
    dC = np.diff(C_samples)

    # Count significant drops
    drops = np.sum(dC < -drop_threshold)

    return int(drops)


def count_clusters(theta: NDArray[np.float64], n_bins: int = 36) -> int:
    """
    Count the number of phase clusters using histogram binning.

    A simple, deterministic method that bins phases and counts
    contiguous groups of occupied bins.

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Phases of all oscillators (should be wrapped to [0, 2π)).
    n_bins : int, optional
        Number of bins for histogram (default 36, i.e., 10° bins).

    Returns
    -------
    n_clusters : int
        Number of distinct phase clusters.
    """
    if len(theta) == 0:
        return 0

    # Wrap phases to [0, 2π)
    theta_wrapped = np.mod(theta, 2 * np.pi)

    # Create histogram
    counts, _ = np.histogram(theta_wrapped, bins=n_bins, range=(0, 2 * np.pi))

    # Find occupied bins (at least one oscillator)
    occupied = counts > 0

    # Count contiguous groups (handling wraparound)
    n_clusters = 0
    in_cluster = False

    # Handle circular boundary: check if first and last bins are part of same cluster
    starts_occupied = occupied[0]
    ends_occupied = occupied[-1]

    for i, occ in enumerate(occupied):
        if occ and not in_cluster:
            n_clusters += 1
            in_cluster = True
        elif not occ:
            in_cluster = False

    # If both ends are occupied, they might be part of the same cluster
    if starts_occupied and ends_occupied and n_clusters > 1:
        # Check if there's a gap between them
        # If no gap, merge by subtracting 1
        first_gap = np.argmax(~occupied)
        if first_gap == 0:
            # No occupied bins at start after all
            pass
        else:
            # There are occupied bins at start
            last_gap = n_bins - 1 - np.argmax(~occupied[::-1])
            if last_gap == n_bins - 1:
                # No gap at end, they're connected
                n_clusters -= 1

    return max(1, n_clusters)


def detect_inversion(
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    omega_d: float,
    t: float,
    threshold: float = np.pi / 2,
) -> bool:
    """
    Detect if the mean phase is inverted relative to the drive.

    Checks if theta_i - phi_i is inverted relative to omega_d * t.

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Final phases of all oscillators.
    phi : ndarray of shape (N,)
        Drive phase offsets.
    omega_d : float
        Drive frequency.
    t : float
        Current time.
    threshold : float, optional
        Threshold for inversion detection (default π/2).

    Returns
    -------
    inverted : bool
        True if mean phase is near π from expected drive phase.
    """
    # Calculate phase relative to drive offset
    # We want to check if (theta - phi) is locked to omega_d * t
    # or omega_d * t + pi

    # Compute order parameter of adjusted phases
    # Z = (1/N) * sum(exp(i * (theta_j - phi_j)))
    complex_phases = np.exp(1j * (theta - phi))
    Z = np.mean(complex_phases)
    mean_phase_adjusted = np.angle(Z)

    # Expected phase is just the drive phase accumulation
    expected_phase = omega_d * t

    # Phase difference
    phase_diff = np.mod(mean_phase_adjusted - expected_phase + np.pi, 2 * np.pi) - np.pi

    # Check if near π (inverted)
    return bool(abs(abs(phase_diff) - np.pi) < threshold)


def compute_phase_velocity(
    C_samples: NDArray[np.float64], t_samples: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the rate of change of coherence.

    Parameters
    ----------
    C_samples : ndarray of shape (n_samples,)
        Coherence values over time.
    t_samples : ndarray of shape (n_samples,)
        Time points.

    Returns
    -------
    dC_dt : ndarray of shape (n_samples-1,)
        Rate of change of coherence.
    """
    if len(C_samples) < 2:
        return np.array([])

    dC = np.diff(C_samples)
    dt = np.diff(t_samples)

    # Avoid division by zero
    dt = np.where(dt == 0, 1e-10, dt)

    return dC / dt


def is_stable(stats: CoherenceStats, C_threshold: float) -> bool:
    """
    Determine if the system is stable based on coherence threshold.

    Parameters
    ----------
    stats : CoherenceStats
        Computed coherence statistics.
    C_threshold : float
        Stability threshold.

    Returns
    -------
    stable : bool
        True if C_mean >= C_threshold.
    """
    return stats.C_mean >= C_threshold
