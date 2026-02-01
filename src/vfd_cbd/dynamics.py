"""
Dynamics simulation for coupled phase oscillators.

Implements the driven Kuramoto-like model:
    dθ_i/dt = ω_i + Σ_j K_ij * sin(θ_j - θ_i) + D_i * sin(ω_d t + φ_i - θ_i)

Uses fixed-step RK4 integration for bitwise reproducibility.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    theta_final: NDArray[np.float64]  # Final phase state (N,)
    C_samples: NDArray[np.float64]    # Sampled coherence values over time
    t_samples: NDArray[np.float64]    # Time points for samples
    n_steps: int                       # Total number of steps taken


def simulate(
    N: int,
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    omega_d: float,
    D: NDArray[np.float64],
    phi: NDArray[np.float64],
    dt: float,
    T: float,
    sample_every: int = 10,
    theta_init: Optional[NDArray[np.float64]] = None,
    seed: Optional[int] = None
) -> SimulationResult:
    """
    Simulate the coupled oscillator dynamics.

    Uses fixed-step RK4 integration for determinism.

    Parameters
    ----------
    N : int
        Number of oscillators.
    K : ndarray of shape (N, N)
        Coupling matrix.
    omega : ndarray of shape (N,)
        Natural frequencies of oscillators.
    omega_d : float
        Drive frequency.
    D : ndarray of shape (N,)
        Drive coupling strengths.
    phi : ndarray of shape (N,)
        Phase offsets for drive.
    dt : float
        Time step for integration.
    T : float
        Total simulation time.
    sample_every : int, optional
        Sample coherence every N steps (default 10).
    theta_init : ndarray of shape (N,), optional
        Initial phases. If None, random uniform [0, 2π).
    seed : int, optional
        Random seed for initial phases if theta_init is None.

    Returns
    -------
    result : SimulationResult
        Simulation results including final phases and coherence samples.
    """
    # Set up random state for initial conditions
    if theta_init is None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, N)
    else:
        theta = theta_init.copy()

    # Calculate number of steps
    n_steps = int(np.ceil(T / dt))
    n_samples = (n_steps // sample_every) + 1

    # Preallocate sample arrays
    C_samples = np.zeros(n_samples, dtype=np.float64)
    t_samples = np.zeros(n_samples, dtype=np.float64)

    # Initial coherence
    C_samples[0] = compute_coherence(theta)
    t_samples[0] = 0.0

    sample_idx = 1
    t = 0.0

    # Main integration loop
    for step in range(1, n_steps + 1):
        theta = rk4_step(theta, t, dt, K, omega, omega_d, D, phi)
        t += dt

        # Sample coherence periodically
        if step % sample_every == 0 and sample_idx < n_samples:
            C_samples[sample_idx] = compute_coherence(theta)
            t_samples[sample_idx] = t
            sample_idx += 1

    # Ensure final state is sampled
    if sample_idx < n_samples:
        C_samples[sample_idx] = compute_coherence(theta)
        t_samples[sample_idx] = t

    return SimulationResult(
        theta_final=wrap_phase(theta),
        C_samples=C_samples[:sample_idx + 1] if sample_idx < n_samples else C_samples,
        t_samples=t_samples[:sample_idx + 1] if sample_idx < n_samples else t_samples,
        n_steps=n_steps
    )


def compute_derivatives(
    theta: NDArray[np.float64],
    t: float,
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    omega_d: float,
    D: NDArray[np.float64],
    phi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute dθ/dt for all oscillators.

    dθ_i/dt = ω_i + Σ_j K_ij * sin(θ_j - θ_i) + D_i * sin(ω_d t + φ_i - θ_i)

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Current phases.
    t : float
        Current time.
    K : ndarray of shape (N, N)
        Coupling matrix.
    omega : ndarray of shape (N,)
        Natural frequencies.
    omega_d : float
        Drive frequency.
    D : ndarray of shape (N,)
        Drive coupling strengths.
    phi : ndarray of shape (N,)
        Phase offsets for drive.

    Returns
    -------
    dtheta_dt : ndarray of shape (N,)
        Time derivatives of phases.
    """
    N = len(theta)

    # Coupling term: Σ_j K_ij * sin(θ_j - θ_i)
    # theta_diff[i, j] = theta[j] - theta[i]
    theta_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling = np.sum(K * np.sin(theta_diff), axis=1)

    # Drive term: D_i * sin(ω_d t + φ_i - θ_i)
    drive_phase = omega_d * t + phi - theta
    drive = D * np.sin(drive_phase)

    # Total derivative
    dtheta_dt = omega + coupling + drive

    return dtheta_dt


def rk4_step(
    theta: NDArray[np.float64],
    t: float,
    dt: float,
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    omega_d: float,
    D: NDArray[np.float64],
    phi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Perform a single RK4 integration step.

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Current phases.
    t : float
        Current time.
    dt : float
        Time step.
    K : ndarray of shape (N, N)
        Coupling matrix.
    omega : ndarray of shape (N,)
        Natural frequencies.
    omega_d : float
        Drive frequency.
    D : ndarray of shape (N,)
        Drive coupling strengths.
    phi : ndarray of shape (N,)
        Phase offsets for drive.

    Returns
    -------
    theta_new : ndarray of shape (N,)
        Updated phases after RK4 step.
    """
    k1 = compute_derivatives(theta, t, K, omega, omega_d, D, phi)
    k2 = compute_derivatives(theta + 0.5 * dt * k1, t + 0.5 * dt, K, omega, omega_d, D, phi)
    k3 = compute_derivatives(theta + 0.5 * dt * k2, t + 0.5 * dt, K, omega, omega_d, D, phi)
    k4 = compute_derivatives(theta + dt * k3, t + dt, K, omega, omega_d, D, phi)

    theta_new = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return theta_new


def compute_coherence(theta: NDArray[np.float64]) -> float:
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
    """
    N = len(theta)
    complex_phases = np.exp(1j * theta)
    C = np.abs(np.mean(complex_phases))
    return C


def wrap_phase(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Wrap phases to [0, 2π).

    Parameters
    ----------
    theta : ndarray of shape (N,)
        Unwrapped phases.

    Returns
    -------
    theta_wrapped : ndarray of shape (N,)
        Phases wrapped to [0, 2π).
    """
    return np.mod(theta, 2 * np.pi)


def get_phase_offsets(
    N: int,
    phase_mode: Literal["single", "dual", "custom"],
    phase_custom: Optional[list[float]] = None
) -> NDArray[np.float64]:
    """
    Get phase offset pattern for the drive.

    Parameters
    ----------
    N : int
        Number of oscillators.
    phase_mode : {"single", "dual", "custom"}
        Phase offset pattern mode.
    phase_custom : list of float, optional
        Custom phase offsets (required if phase_mode is "custom").

    Returns
    -------
    phi : ndarray of shape (N,)
        Phase offsets for each oscillator.
    """
    if phase_mode == "single":
        return np.zeros(N, dtype=np.float64)
    elif phase_mode == "dual":
        phi = np.zeros(N, dtype=np.float64)
        phi[N // 2:] = np.pi
        return phi
    elif phase_mode == "custom":
        if phase_custom is None:
            raise ValueError("phase_custom must be provided for custom mode")
        if len(phase_custom) != N:
            raise ValueError(f"phase_custom length ({len(phase_custom)}) must match N ({N})")
        return np.array(phase_custom, dtype=np.float64)
    else:
        raise ValueError(f"Unknown phase_mode: {phase_mode}")


def get_natural_frequencies(
    N: int,
    f0_hz: float,
    detuning: float
) -> NDArray[np.float64]:
    """
    Get natural frequencies for all oscillators.

    For MVP, all oscillators have the same detuning from the base frequency.

    Parameters
    ----------
    N : int
        Number of oscillators.
    f0_hz : float
        Base frequency in Hz.
    detuning : float
        Fractional detuning (Δf/f0).

    Returns
    -------
    omega : ndarray of shape (N,)
        Angular frequencies (rad/s) for each oscillator.
    """
    f = f0_hz * (1 + detuning)
    omega = 2 * np.pi * f * np.ones(N, dtype=np.float64)
    return omega


def get_drive_coupling(
    N: int,
    A0: float
) -> NDArray[np.float64]:
    """
    Get drive coupling strengths for all oscillators.

    For MVP, uniform drive coupling.

    Parameters
    ----------
    N : int
        Number of oscillators.
    A0 : float
        Drive amplitude.

    Returns
    -------
    D : ndarray of shape (N,)
        Drive coupling strength for each oscillator.
    """
    return A0 * np.ones(N, dtype=np.float64)
