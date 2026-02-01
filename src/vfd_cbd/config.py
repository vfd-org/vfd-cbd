"""
Configuration management for CBD simulations.

Handles loading, validation, and defaulting of YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Literal
import yaml
import copy


@dataclass
class RunConfig:
    """Run-level configuration."""
    seed: int = 42
    out_dir: str = "out"
    run_name: Optional[str] = None


@dataclass
class SystemConfig:
    """System geometry and coupling configuration."""
    N: int = 64
    geometry: Literal["ring", "circle_decay"] = "ring"
    radius: float = 1.0
    neighbor_k: int = 2  # Number of neighbors on each side for ring
    lambda_decay: float = 0.5  # Decay length for circle_decay
    cutoff: float = 3.0  # Cutoff in units of lambda for circle_decay


@dataclass
class DriveConfig:
    """Drive configuration."""
    f0_hz: float = 100.0
    A0: float = 1.0  # Drive amplitude
    phase_mode: Literal["single", "dual", "custom"] = "single"
    phase_custom: Optional[list[float]] = None


@dataclass
class LossConfig:
    """Loss/damping configuration."""
    gamma: float = 0.1


@dataclass
class SweepConfig:
    """Parameter sweep configuration."""
    df_min: float = -0.1
    df_max: float = 0.1
    n_df: int = 25
    k_min: float = 0.1
    k_max: float = 2.0
    n_k: int = 25


@dataclass
class SimConfig:
    """Simulation parameters."""
    dt: float = 0.001
    T: float = 10.0
    sample_every: int = 10
    window_frac: float = 0.2


@dataclass
class ThresholdConfig:
    """Thresholds for stability and mode classification."""
    C_threshold: float = 0.7
    slip_threshold: int = 5
    cluster_threshold: int = 2


@dataclass
class Config:
    """Complete configuration for a CBD simulation."""
    run: RunConfig = field(default_factory=RunConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        return {
            "run": {
                "seed": self.run.seed,
                "out_dir": self.run.out_dir,
                "run_name": self.run.run_name,
            },
            "system": {
                "N": self.system.N,
                "geometry": self.system.geometry,
                "radius": self.system.radius,
                "neighbor_k": self.system.neighbor_k,
                "lambda_decay": self.system.lambda_decay,
                "cutoff": self.system.cutoff,
            },
            "drive": {
                "f0_hz": self.drive.f0_hz,
                "A0": self.drive.A0,
                "phase_mode": self.drive.phase_mode,
                "phase_custom": self.drive.phase_custom,
            },
            "loss": {
                "gamma": self.loss.gamma,
            },
            "sweep": {
                "df_min": self.sweep.df_min,
                "df_max": self.sweep.df_max,
                "n_df": self.sweep.n_df,
                "k_min": self.sweep.k_min,
                "k_max": self.sweep.k_max,
                "n_k": self.sweep.n_k,
            },
            "sim": {
                "dt": self.sim.dt,
                "T": self.sim.T,
                "sample_every": self.sim.sample_every,
                "window_frac": self.sim.window_frac,
            },
            "thresholds": {
                "C_threshold": self.thresholds.C_threshold,
                "slip_threshold": self.thresholds.slip_threshold,
                "cluster_threshold": self.thresholds.cluster_threshold,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create Config from nested dictionary."""
        config = cls()

        if "run" in d:
            for key, value in d["run"].items():
                if hasattr(config.run, key):
                    setattr(config.run, key, value)

        if "system" in d:
            for key, value in d["system"].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)

        if "drive" in d:
            for key, value in d["drive"].items():
                if hasattr(config.drive, key):
                    setattr(config.drive, key, value)

        if "loss" in d:
            for key, value in d["loss"].items():
                if hasattr(config.loss, key):
                    setattr(config.loss, key, value)

        if "sweep" in d:
            for key, value in d["sweep"].items():
                if hasattr(config.sweep, key):
                    setattr(config.sweep, key, value)

        if "sim" in d:
            for key, value in d["sim"].items():
                if hasattr(config.sim, key):
                    setattr(config.sim, key, value)

        if "thresholds" in d:
            for key, value in d["thresholds"].items():
                if hasattr(config.thresholds, key):
                    setattr(config.thresholds, key, value)

        return config


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    Config
        Loaded and validated configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    config = Config.from_dict(data)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Parameters
    ----------
    config : Config
        Configuration to validate.

    Raises
    ------
    ValueError
        If any configuration value is invalid.
    """
    # System validation
    if config.system.N < 2:
        raise ValueError("N must be at least 2")
    if config.system.geometry not in ("ring", "circle_decay"):
        raise ValueError(f"Unknown geometry: {config.system.geometry}")
    if config.system.radius <= 0:
        raise ValueError("radius must be positive")
    if config.system.neighbor_k < 1:
        raise ValueError("neighbor_k must be at least 1")
    if config.system.lambda_decay <= 0:
        raise ValueError("lambda_decay must be positive")
    if config.system.cutoff <= 0:
        raise ValueError("cutoff must be positive")

    # Drive validation
    if config.drive.f0_hz <= 0:
        raise ValueError("f0_hz must be positive")
    if config.drive.A0 < 0:
        raise ValueError("A0 must be non-negative")
    if config.drive.phase_mode not in ("single", "dual", "custom"):
        raise ValueError(f"Unknown phase_mode: {config.drive.phase_mode}")
    if config.drive.phase_mode == "custom":
        if config.drive.phase_custom is None:
            raise ValueError("phase_custom must be provided for custom phase_mode")
        if len(config.drive.phase_custom) != config.system.N:
            raise ValueError(
                f"phase_custom length ({len(config.drive.phase_custom)}) "
                f"must match N ({config.system.N})"
            )

    # Loss validation
    if config.loss.gamma <= 0:
        raise ValueError("gamma must be positive")

    # Sweep validation
    if config.sweep.n_df < 1:
        raise ValueError("n_df must be at least 1")
    if config.sweep.n_k < 1:
        raise ValueError("n_k must be at least 1")
    if config.sweep.k_min <= 0:
        raise ValueError("k_min must be positive")
    if config.sweep.k_max <= config.sweep.k_min:
        raise ValueError("k_max must be greater than k_min")

    # Sim validation
    if config.sim.dt <= 0:
        raise ValueError("dt must be positive")
    if config.sim.T <= 0:
        raise ValueError("T must be positive")
    if config.sim.sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    if not 0 < config.sim.window_frac <= 1:
        raise ValueError("window_frac must be in (0, 1]")

    # Threshold validation
    if not 0 <= config.thresholds.C_threshold <= 1:
        raise ValueError("C_threshold must be in [0, 1]")
    if config.thresholds.slip_threshold < 1:
        raise ValueError("slip_threshold must be at least 1")
    if config.thresholds.cluster_threshold < 2:
        raise ValueError("cluster_threshold must be at least 2")


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Parameters
    ----------
    config : Config
        Configuration to save.
    path : str or Path
        Path to save YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
