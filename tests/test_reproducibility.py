"""
Tests for reproducibility of simulations.

Verifies that running the same configuration with the same seed
produces identical results.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_cbd.config import Config
from vfd_cbd.sweep import run_sweep
from vfd_cbd.io import compute_config_hash


class TestReproducibility:
    """Tests for simulation reproducibility."""

    @pytest.fixture
    def tiny_config(self):
        """Create a tiny configuration for fast testing."""
        config = Config()
        config.run.seed = 12345
        config.system.N = 16
        config.system.geometry = "ring"
        config.system.neighbor_k = 1
        config.sweep.n_df = 3
        config.sweep.n_k = 3
        config.sweep.df_min = -0.05
        config.sweep.df_max = 0.05
        config.sweep.k_min = 0.5
        config.sweep.k_max = 1.5
        config.sim.T = 2.0
        config.sim.dt = 0.01
        config.sim.sample_every = 5
        return config

    def test_identical_results_same_seed(self, tiny_config):
        """Two runs with same config and seed should produce identical results."""
        # Run sweep twice
        result1 = run_sweep(tiny_config)
        result2 = run_sweep(tiny_config)

        # Check C_mean arrays are identical
        np.testing.assert_array_equal(
            result1.C_mean, result2.C_mean,
            err_msg="C_mean arrays differ between runs"
        )

        # Check C_min arrays
        np.testing.assert_array_equal(
            result1.C_min, result2.C_min,
            err_msg="C_min arrays differ between runs"
        )

        # Check stable_mask
        np.testing.assert_array_equal(
            result1.stable_mask, result2.stable_mask,
            err_msg="stable_mask arrays differ between runs"
        )

        # Check mode_ids
        np.testing.assert_array_equal(
            result1.mode_ids, result2.mode_ids,
            err_msg="mode_ids arrays differ between runs"
        )

    def test_config_hash_stability(self, tiny_config):
        """Config hash should be stable across computations."""
        hash1 = compute_config_hash(tiny_config)
        hash2 = compute_config_hash(tiny_config)

        assert hash1 == hash2, "Config hash should be deterministic"

    def test_different_seeds_different_results(self, tiny_config):
        """Different seeds should produce different results."""
        config1 = tiny_config
        config1.run.seed = 111

        config2 = Config()
        config2.run.seed = 222
        config2.system.N = tiny_config.system.N
        config2.system.geometry = tiny_config.system.geometry
        config2.system.neighbor_k = tiny_config.system.neighbor_k
        config2.sweep.n_df = tiny_config.sweep.n_df
        config2.sweep.n_k = tiny_config.sweep.n_k
        config2.sweep.df_min = tiny_config.sweep.df_min
        config2.sweep.df_max = tiny_config.sweep.df_max
        config2.sweep.k_min = tiny_config.sweep.k_min
        config2.sweep.k_max = tiny_config.sweep.k_max
        config2.sim.T = tiny_config.sim.T
        config2.sim.dt = tiny_config.sim.dt
        config2.sim.sample_every = tiny_config.sim.sample_every

        result1 = run_sweep(config1)
        result2 = run_sweep(config2)

        # Results should differ (at least some values)
        # Note: They might not differ at every point, but overall should be different
        assert not np.allclose(result1.C_mean, result2.C_mean, rtol=1e-5), \
            "Different seeds should produce different results"

    def test_hash_changes_with_config(self, tiny_config):
        """Different configs should produce different hashes."""
        hash1 = compute_config_hash(tiny_config)

        # Modify config
        tiny_config.run.seed = 99999
        hash2 = compute_config_hash(tiny_config)

        assert hash1 != hash2, "Hash should change when config changes"

    def test_result_hash_matches_config(self, tiny_config):
        """Result's config_hash should match computed hash."""
        result = run_sweep(tiny_config)
        expected_hash = compute_config_hash(tiny_config)

        assert result.config_hash == expected_hash, \
            "Result config_hash should match computed hash"


class TestDynamicsReproducibility:
    """Tests for dynamics simulation reproducibility."""

    def test_rk4_determinism(self):
        """RK4 integration should be deterministic."""
        from vfd_cbd.dynamics import simulate
        from vfd_cbd.coupling import create_coupling_matrix

        N = 8
        K = create_coupling_matrix(N, k_strength=0.5, geometry="ring", neighbor_k=1)
        omega = np.full(N, 2 * np.pi * 100)
        omega_d = 2 * np.pi * 100
        D = np.ones(N)
        phi = np.zeros(N)

        # Run twice with same seed
        result1 = simulate(N, K, omega, omega_d, D, phi, dt=0.01, T=1.0, seed=42)
        result2 = simulate(N, K, omega, omega_d, D, phi, dt=0.01, T=1.0, seed=42)

        # Final phases should be identical
        np.testing.assert_array_equal(
            result1.theta_final, result2.theta_final,
            err_msg="Final phases differ between runs"
        )

        # Coherence samples should be identical
        np.testing.assert_array_equal(
            result1.C_samples, result2.C_samples,
            err_msg="Coherence samples differ between runs"
        )

    def test_different_initial_conditions(self):
        """Different seeds should give different initial conditions."""
        from vfd_cbd.dynamics import simulate
        from vfd_cbd.coupling import create_coupling_matrix

        N = 8
        K = create_coupling_matrix(N, k_strength=0.5, geometry="ring", neighbor_k=1)
        omega = np.full(N, 2 * np.pi * 100)
        omega_d = 2 * np.pi * 100
        D = np.ones(N)
        phi = np.zeros(N)

        result1 = simulate(N, K, omega, omega_d, D, phi, dt=0.01, T=0.5, seed=1)
        result2 = simulate(N, K, omega, omega_d, D, phi, dt=0.01, T=0.5, seed=2)

        # Should have different trajectories
        assert not np.allclose(result1.theta_final, result2.theta_final), \
            "Different seeds should give different results"
