"""
GPU-accelerated parameter sweep using JAX.
"""

import time
from datetime import datetime
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .config import Config
from .coupling import create_coupling_matrix
from .dynamics import get_drive_coupling, get_phase_offsets
from .sweep import SweepResult
from .metrics import CoherenceStats
from .modes import ModeType, classify_mode

# -----------------------------------------------------------------------------
# JAX-compatible dynamics
# -----------------------------------------------------------------------------


@jax.jit
def compute_derivatives_jax(
    theta: jnp.ndarray,
    t: float,
    K: jnp.ndarray,
    omega: jnp.ndarray,
    omega_d: float,
    D: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute dθ/dt for all oscillators (JAX version).
    """
    # Coupling term: Σ_j K_ij * sin(θ_j - θ_i)
    # theta_diff[i, j] = theta[j] - theta[i]
    theta_diff = theta[None, :] - theta[:, None]
    coupling = jnp.sum(K * jnp.sin(theta_diff), axis=1)

    # Drive term: D_i * sin(ω_d t + φ_i - θ_i)
    drive_phase = omega_d * t + phi - theta
    drive = D * jnp.sin(drive_phase)

    # Total derivative
    return omega + coupling + drive


@jax.jit
def rk4_step_jax(
    theta: jnp.ndarray,
    t: float,
    dt: float,
    K: jnp.ndarray,
    omega: jnp.ndarray,
    omega_d: float,
    D: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Perform a single RK4 integration step (JAX version).
    """
    k1 = compute_derivatives_jax(theta, t, K, omega, omega_d, D, phi)
    k2 = compute_derivatives_jax(
        theta + 0.5 * dt * k1, t + 0.5 * dt, K, omega, omega_d, D, phi
    )
    k3 = compute_derivatives_jax(
        theta + 0.5 * dt * k2, t + 0.5 * dt, K, omega, omega_d, D, phi
    )
    k4 = compute_derivatives_jax(theta + dt * k3, t + dt, K, omega, omega_d, D, phi)

    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -----------------------------------------------------------------------------
# Single simulation run
# -----------------------------------------------------------------------------


def run_simulation_jax(
    seed: int,
    N: int,
    K_unit: jnp.ndarray,  # Pre-computed K with k=1
    omega_base: jnp.ndarray,  # Base natural frequencies
    D: jnp.ndarray,
    phi: jnp.ndarray,
    k_val: float,  # scalar parameter from sweep
    df_val: float,  # scalar parameter from sweep
    gamma: float,
    f0_hz: float,
    dt: float,
    n_steps: int,
    sample_every: int,
    n_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run a single simulation for a parameter point (JAX-traceable).
    Returns (theta_final, C_samples).
    """
    # 1. Parameter Setup
    k_strength = k_val * gamma
    K = K_unit * k_strength

    omega_d = 2 * jnp.pi * f0_hz * (1.0 + df_val)

    # Oscillators stay at base frequency in this model
    omega = omega_base

    # 2. Initialization
    key = jax.random.PRNGKey(seed)
    theta_init = jax.random.uniform(key, shape=(N,), minval=0.0, maxval=2 * jnp.pi)

    # 3. Time Loop using lax.scan

    # State: (theta, time_index, sample_idx)
    # But for scan, we typically carry the state and stack outputs.
    # To implement "sample_every", we can just output every step and slice later,
    # or use a modulo check if we only care about the carry.
    # For GPU memory efficiency with large T, we should only store samples.
    # However, lax.scan outputs the stack of all steps.
    # A better approach for "sample_every":
    # Use nested loops or just scan over n_steps and use `jax.lax.cond` to update a sampling buffer?
    # Actually, `lax.scan` is good. If we want to subsample, we can reshape the loop.
    # E.g. Outer loop `n_samples`, inner loop `sample_every`.

    def inner_step(carry, _):
        theta, t = carry
        theta_new = rk4_step_jax(theta, t, dt, K, omega, omega_d, D, phi)
        return (theta_new, t + dt), None

    def outer_step(carry, _):
        # Run `sample_every` steps
        (theta, t), _ = jax.lax.scan(inner_step, carry, None, length=sample_every)

        # Compute coherence
        complex_phases = jnp.exp(1j * theta)
        C = jnp.abs(jnp.mean(complex_phases))

        return (theta, t), C

    # Initial run
    init_state = (theta_init, 0.0)

    # First sample (t=0)
    C_0 = jnp.abs(jnp.mean(jnp.exp(1j * theta_init)))

    # Run the loop
    # We need (n_samples - 1) more samples
    final_state, C_rest = jax.lax.scan(
        outer_step, init_state, None, length=n_samples - 1
    )

    theta_final = final_state[0]

    # Combine samples
    C_samples = jnp.concatenate([jnp.array([C_0]), C_rest])

    return theta_final, C_samples


# -----------------------------------------------------------------------------
# Batch Sweep
# -----------------------------------------------------------------------------


def run_sweep_gpu(config: Config) -> SweepResult:
    """
    Run the parameter sweep on GPU.
    """
    from .io import compute_config_hash

    print(f"JAX backend: {jax.devices()[0]}")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract parameters
    N = config.system.N
    geometry = config.system.geometry
    f0_hz = config.drive.f0_hz
    A0 = config.drive.A0
    gamma = config.loss.gamma

    dt = config.sim.dt
    T = config.sim.T
    sample_every = config.sim.sample_every

    n_steps = int(np.ceil(T / dt))
    n_samples = (n_steps // sample_every) + 1

    # Grid
    n_df = config.sweep.n_df
    n_k = config.sweep.n_k
    df_values = np.linspace(config.sweep.df_min, config.sweep.df_max, n_df)
    k_values = np.linspace(config.sweep.k_min, config.sweep.k_max, n_k)

    # Create meshgrid for vectorized mapping
    # We want shape (n_k, n_df)
    KV, DFV = np.meshgrid(k_values, df_values, indexing="ij")

    # Flatten for vmap
    k_flat = jnp.array(KV.flatten())
    df_flat = jnp.array(DFV.flatten())

    # Pre-compute constant structures on CPU then move to GPU
    # 1. Base coupling matrix (k=1)
    K_unit_np = create_coupling_matrix(
        N=N,
        k_strength=1.0,
        geometry=geometry,
        neighbor_k=config.system.neighbor_k,
        radius=config.system.radius,
        lambda_decay=config.system.lambda_decay,
        cutoff=config.system.cutoff,
    )
    K_unit = jnp.array(K_unit_np)

    # 2. Drive coupling
    D = jnp.array(get_drive_coupling(N, A0))

    # 3. Phase offsets
    phi = jnp.array(
        get_phase_offsets(N, config.drive.phase_mode, config.drive.phase_custom)
    )

    # 4. Base Natural frequencies
    omega_base = jnp.array(2 * np.pi * f0_hz * np.ones(N))

    # 5. Seeds
    # Unique seed per point: base_seed + index
    base_seed = config.run.seed
    seeds = jnp.arange(base_seed, base_seed + len(k_flat))

    # Define the vmapped function
    # We fix the static config parameters using closure or partial,
    # but passing them as args to the jitted function is fine too if they are arrays.
    # Scalars like dt, gamma should be fine.

    run_sim_vmap = jax.vmap(
        lambda s, k, df: run_simulation_jax(
            s,
            N,
            K_unit,
            omega_base,
            D,
            phi,
            k,
            df,
            gamma,
            f0_hz,
            dt,
            n_steps,
            sample_every,
            n_samples,
        )
    )

    total_points = len(k_flat)
    print(f"Compiling and running sweep on {total_points} points...")

    # Batching for progress reporting
    BATCH_SIZE = 5000
    n_batches = int(np.ceil(total_points / BATCH_SIZE))

    theta_finals_list = []
    C_samples_list = []

    print(f"Processing in {n_batches} batches of size {BATCH_SIZE}...")

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_points)

        # Slice inputs
        seeds_batch = seeds[start_idx:end_idx]
        k_batch = k_flat[start_idx:end_idx]
        df_batch = df_flat[start_idx:end_idx]

        # Run batch
        # Returns (batch_size, N), (batch_size, n_samples)
        theta_finals_b, C_samples_b = run_sim_vmap(seeds_batch, k_batch, df_batch)

        # Force execution to measure progress
        theta_finals_b.block_until_ready()

        # Store results (move to host to free GPU memory)
        theta_finals_list.append(np.array(theta_finals_b))
        C_samples_list.append(np.array(C_samples_b))

        # Report progress
        percent = (i + 1) / n_batches * 100
        print(f"  Batch {i + 1}/{n_batches} ({percent:.1f}%) complete")

    elapsed = time.time() - start_time
    print(f"Sweep completed in {elapsed:.2f}s")

    # -------------------------------------------------------------------------
    # Post-processing (on CPU)
    # -------------------------------------------------------------------------
    # Concatenate results
    theta_finals_np = np.concatenate(theta_finals_list, axis=0)
    C_samples_np = np.concatenate(C_samples_list, axis=0)

    # Reshape to grid (n_k, n_df, ...)
    theta_finals_grid = theta_finals_np.reshape(n_k, n_df, N)
    C_samples_grid = C_samples_np.reshape(n_k, n_df, n_samples)

    # Preallocate metric arrays
    C_mean_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_min_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_max_arr = np.zeros((n_k, n_df), dtype=np.float64)
    C_std_arr = np.zeros((n_k, n_df), dtype=np.float64)
    stable_mask = np.zeros((n_k, n_df), dtype=np.bool_)
    mode_ids = np.zeros((n_k, n_df), dtype=np.int32)
    phase_slip_counts = np.zeros((n_k, n_df), dtype=np.int32)
    cluster_counts = np.zeros((n_k, n_df), dtype=np.int32)

    window_frac = config.sim.window_frac

    # Post-process metrics (can also be JIT-ed potentially, but CPU is fine for metrics usually)
    # Using existing metric functions from .metrics and .modes

    from .metrics import (
        compute_coherence_stats,
        count_phase_slips_from_coherence,
        count_clusters,
        detect_inversion,
    )
    from .dynamics import wrap_phase

    print("Computing metrics...")

    # We iterate to re-use the existing logic.
    # Parallelizing this on CPU would be easy too, but let's stick to simple loops first.
    # It's fast enough relative to simulation.

    total_points = n_k * n_df
    for i_k in range(n_k):
        for i_df in range(n_df):
            C_series = C_samples_grid[i_k, i_df]
            theta_f = wrap_phase(theta_finals_grid[i_k, i_df])

            # Reconstruct params for inversion check
            omega_d_val = 2 * np.pi * f0_hz * (1 + df_values[i_df])

            stats = compute_coherence_stats(C_series, window_frac)
            slip_count = count_phase_slips_from_coherence(C_series)
            clusters = count_clusters(theta_f)

            # For inversion, we need phi (numpy version)
            # We can grab it from JAX array or recompute
            phi_np = np.array(phi)

            inversion = detect_inversion(theta_f, phi_np, omega_d_val, T)

            mode = classify_mode(
                stats=stats,
                phase_slip_count=slip_count,
                cluster_count=clusters,
                inversion_flag=inversion,
                C_threshold=config.thresholds.C_threshold,
                slip_threshold=config.thresholds.slip_threshold,
                cluster_threshold=config.thresholds.cluster_threshold,
            )

            C_mean_arr[i_k, i_df] = stats.C_mean
            C_min_arr[i_k, i_df] = stats.C_min
            C_max_arr[i_k, i_df] = stats.C_max
            C_std_arr[i_k, i_df] = stats.C_std
            stable_mask[i_k, i_df] = stats.C_mean >= config.thresholds.C_threshold
            mode_ids[i_k, i_df] = int(mode)
            phase_slip_counts[i_k, i_df] = slip_count
            cluster_counts[i_k, i_df] = clusters

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
        total_points=total_points,
    )
