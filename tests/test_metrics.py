"""
Tests for coherence metrics.

Verifies that the Kuramoto order parameter and related metrics
behave correctly for known cases.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_cbd.metrics import (
    kuramoto_order_parameter,
    kuramoto_complex_order,
    compute_coherence_stats,
    count_clusters,
    is_stable,
    CoherenceStats
)


class TestKuramotoOrderParameter:
    """Tests for the Kuramoto order parameter calculation."""

    def test_identical_phases_high_coherence(self):
        """All oscillators in phase should give coherence near 1."""
        # All phases at 0
        theta = np.zeros(100)
        C = kuramoto_order_parameter(theta)
        assert C == pytest.approx(1.0, abs=1e-10)

        # All phases at π/2
        theta = np.full(100, np.pi / 2)
        C = kuramoto_order_parameter(theta)
        assert C == pytest.approx(1.0, abs=1e-10)

        # All phases at arbitrary value
        theta = np.full(100, 2.5)
        C = kuramoto_order_parameter(theta)
        assert C == pytest.approx(1.0, abs=1e-10)

    def test_uniform_random_phases_low_coherence(self):
        """Uniformly random phases should give low coherence for large N."""
        # With fixed seed for reproducibility
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, 64)
        C = kuramoto_order_parameter(theta)

        # For N=64 random phases, expected C should be small
        # C ~ 1/sqrt(N) ~ 0.125 for random phases
        assert C < 0.3, f"Expected low coherence for random phases, got {C}"

    def test_uniform_random_phases_large_N(self):
        """For very large N, random phases should give very low coherence."""
        rng = np.random.default_rng(123)
        theta = rng.uniform(0, 2 * np.pi, 1000)
        C = kuramoto_order_parameter(theta)

        # For N=1000, expected C ~ 1/sqrt(1000) ~ 0.032
        assert C < 0.1, f"Expected very low coherence for N=1000 random phases, got {C}"

    def test_two_clusters_half_coherence(self):
        """Two opposite clusters should give intermediate coherence."""
        # Half at 0, half at π
        theta = np.concatenate([np.zeros(50), np.full(50, np.pi)])
        C = kuramoto_order_parameter(theta)

        # Two opposite clusters cancel out
        assert C == pytest.approx(0.0, abs=1e-10)

    def test_two_clusters_partial(self):
        """Two clusters at 90 degrees should give specific coherence."""
        # Half at 0, half at π/2
        theta = np.concatenate([np.zeros(50), np.full(50, np.pi / 2)])
        C = kuramoto_order_parameter(theta)

        # Expected: |0.5 + 0.5*i| = sqrt(0.5) ≈ 0.707
        expected = np.sqrt(0.5)
        assert C == pytest.approx(expected, abs=1e-10)

    def test_empty_array(self):
        """Empty array should return 0."""
        theta = np.array([])
        C = kuramoto_order_parameter(theta)
        assert C == 0.0

    def test_single_oscillator(self):
        """Single oscillator should give coherence of 1."""
        theta = np.array([1.5])
        C = kuramoto_order_parameter(theta)
        assert C == pytest.approx(1.0, abs=1e-10)


class TestComplexOrder:
    """Tests for the complex order parameter."""

    def test_phase_extraction(self):
        """Mean phase should be extractable from complex order."""
        # All phases at π/4
        theta = np.full(100, np.pi / 4)
        Z = kuramoto_complex_order(theta)

        assert np.abs(Z) == pytest.approx(1.0, abs=1e-10)
        assert np.angle(Z) == pytest.approx(np.pi / 4, abs=1e-10)


class TestCoherenceStats:
    """Tests for coherence statistics computation."""

    def test_constant_coherence(self):
        """Constant coherence should give matching stats."""
        C_samples = np.full(100, 0.8)
        stats = compute_coherence_stats(C_samples, window_frac=0.2)

        assert stats.C_mean == pytest.approx(0.8, abs=1e-10)
        assert stats.C_min == pytest.approx(0.8, abs=1e-10)
        assert stats.C_max == pytest.approx(0.8, abs=1e-10)
        assert stats.C_std == pytest.approx(0.0, abs=1e-10)

    def test_window_selection(self):
        """Only the last fraction of samples should be used."""
        # First 80 samples at 0.3, last 20 at 0.9
        C_samples = np.concatenate([np.full(80, 0.3), np.full(20, 0.9)])
        stats = compute_coherence_stats(C_samples, window_frac=0.2)

        # Should only consider last 20%
        assert stats.C_mean == pytest.approx(0.9, abs=1e-10)

    def test_empty_samples(self):
        """Empty samples should return zero stats."""
        C_samples = np.array([])
        stats = compute_coherence_stats(C_samples)

        assert stats.C_mean == 0.0
        assert stats.C_min == 0.0


class TestCountClusters:
    """Tests for cluster counting."""

    def test_single_cluster(self):
        """All phases together should be one cluster."""
        theta = np.zeros(100)
        n = count_clusters(theta)
        assert n == 1

    def test_two_clusters(self):
        """Two distinct phase groups should count as two clusters."""
        # Half at 0, half at π
        theta = np.concatenate([np.zeros(50), np.full(50, np.pi)])
        n = count_clusters(theta)
        assert n == 2

    def test_spread_phases(self):
        """Phases spread around the circle with gaps."""
        # Create three clusters at 0, 2π/3, 4π/3
        theta = np.concatenate([
            np.full(30, 0.0),
            np.full(30, 2 * np.pi / 3),
            np.full(30, 4 * np.pi / 3)
        ])
        n = count_clusters(theta)
        assert n >= 2  # Should detect at least 2 clusters


class TestStability:
    """Tests for stability determination."""

    def test_stable_above_threshold(self):
        """System should be stable when C_mean >= threshold."""
        stats = CoherenceStats(
            C_mean=0.8, C_min=0.7, C_max=0.9, C_std=0.05, C_final=0.85
        )
        assert is_stable(stats, C_threshold=0.7) is True

    def test_unstable_below_threshold(self):
        """System should be unstable when C_mean < threshold."""
        stats = CoherenceStats(
            C_mean=0.5, C_min=0.3, C_max=0.7, C_std=0.1, C_final=0.45
        )
        assert is_stable(stats, C_threshold=0.7) is False

    def test_boundary_case(self):
        """Exactly at threshold should be stable."""
        stats = CoherenceStats(
            C_mean=0.7, C_min=0.6, C_max=0.8, C_std=0.05, C_final=0.7
        )
        assert is_stable(stats, C_threshold=0.7) is True
