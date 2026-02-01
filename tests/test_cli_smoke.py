"""
Smoke tests for the CLI.

Verifies that CLI commands run without errors and produce expected outputs.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCLISmoke:
    """Smoke tests for CLI commands."""

    @pytest.fixture
    def tiny_config_file(self, tmp_path):
        """Create a tiny config file for testing."""
        config_content = """
run:
  seed: 42
  out_dir: "{out_dir}"
  run_name: "test_run"

system:
  N: 8
  geometry: "ring"
  neighbor_k: 1

drive:
  f0_hz: 100.0
  A0: 1.0
  phase_mode: "single"

loss:
  gamma: 0.1

sweep:
  df_min: -0.05
  df_max: 0.05
  n_df: 3
  k_min: 0.5
  k_max: 1.5
  n_k: 3

sim:
  dt: 0.01
  T: 1.0
  sample_every: 5
  window_frac: 0.2

thresholds:
  C_threshold: 0.7
  slip_threshold: 5
  cluster_threshold: 2
"""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content.format(out_dir=str(out_dir)))

        return config_path, out_dir

    def test_sweep_creates_output(self, tiny_config_file):
        """Sweep command should create output folder with required files."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        config_path, out_dir = tiny_config_file
        runner = CliRunner()

        result = runner.invoke(app, ["sweep", "--config", str(config_path), "--quiet"])

        # Check command succeeded
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check output directory was created
        runs = list(out_dir.glob("run_*"))
        assert len(runs) >= 1, "No run folder created"

        run_path = runs[0]

        # Check required files exist
        assert (run_path / "config_resolved.yaml").exists(), "config_resolved.yaml not found"
        assert (run_path / "results.json").exists(), "results.json not found"
        assert (run_path / "results.csv").exists(), "results.csv not found"
        assert (run_path / "plot_phase_diagram.png").exists(), "plot_phase_diagram.png not found"
        assert (run_path / "plot_collapse_boundary.png").exists(), "plot_collapse_boundary.png not found"
        assert (run_path / "plot_mode_map.png").exists(), "plot_mode_map.png not found"

    def test_sweep_no_plots(self, tiny_config_file):
        """Sweep with --no-plots should skip plot generation."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        config_path, out_dir = tiny_config_file
        runner = CliRunner()

        result = runner.invoke(app, ["sweep", "--config", str(config_path), "--no-plots", "--quiet"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        runs = list(out_dir.glob("run_*"))
        assert len(runs) >= 1

        run_path = runs[0]

        # Data files should exist
        assert (run_path / "results.json").exists()
        assert (run_path / "results.csv").exists()

        # Plots should NOT exist
        assert not (run_path / "plot_phase_diagram.png").exists()

    def test_version_command(self):
        """Version command should display version."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_plot_regeneration(self, tiny_config_file):
        """Plot command should regenerate plots from existing run."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        config_path, out_dir = tiny_config_file
        runner = CliRunner()

        # First, create a run without plots
        result = runner.invoke(app, ["sweep", "--config", str(config_path), "--no-plots", "--quiet"])
        assert result.exit_code == 0

        runs = list(out_dir.glob("run_*"))
        run_path = runs[0]

        # Verify no plots yet
        assert not (run_path / "plot_phase_diagram.png").exists()

        # Now regenerate plots
        result = runner.invoke(app, ["plot", "--run", str(run_path)])
        assert result.exit_code == 0

        # Plots should now exist
        assert (run_path / "plot_phase_diagram.png").exists()
        assert (run_path / "plot_collapse_boundary.png").exists()
        assert (run_path / "plot_mode_map.png").exists()

    def test_info_command(self, tiny_config_file):
        """Info command should display run information."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        config_path, out_dir = tiny_config_file
        runner = CliRunner()

        # Create a run
        result = runner.invoke(app, ["sweep", "--config", str(config_path), "--quiet"])
        assert result.exit_code == 0

        runs = list(out_dir.glob("run_*"))
        run_path = runs[0]

        # Get info
        result = runner.invoke(app, ["info", "--run", str(run_path)])
        assert result.exit_code == 0

        # Should contain key information
        assert "Config Hash" in result.output or "Timestamp" in result.output

    def test_export_command(self, tiny_config_file):
        """Export command should export results in specified formats."""
        from typer.testing import CliRunner
        from vfd_cbd.cli import app

        config_path, out_dir = tiny_config_file
        runner = CliRunner()

        # Create a run
        result = runner.invoke(app, ["sweep", "--config", str(config_path), "--quiet"])
        assert result.exit_code == 0

        runs = list(out_dir.glob("run_*"))
        run_path = runs[0]

        # Export
        result = runner.invoke(app, ["export", "--run", str(run_path), "--format", "json,csv"])
        assert result.exit_code == 0

        # Files should exist
        assert (run_path / "results.json").exists()
        assert (run_path / "results.csv").exists()


class TestCLIModuleEntry:
    """Test that the module can be run with python -m."""

    def test_module_import(self):
        """CLI module should be importable."""
        from vfd_cbd import cli
        assert hasattr(cli, 'app')
        assert hasattr(cli, 'main')
