"""
Input/Output functionality for CBD simulations.

Handles saving and loading results, creating run folders,
and computing reproducibility hashes.
"""

import json
import hashlib
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Union, Optional
from datetime import datetime

import pandas as pd
import yaml

from .config import Config, save_config
from .sweep import SweepResult
from .modes import ModeType, mode_to_string


def compute_config_hash(config: Config) -> str:
    """
    Compute a stable hash of the configuration.

    The hash is deterministic and based on the resolved config values.

    Parameters
    ----------
    config : Config
        Configuration to hash.

    Returns
    -------
    hash_str : str
        SHA256 hash of the configuration (first 12 characters).
    """
    config_dict = config.to_dict()
    # Sort keys for determinism
    config_json = json.dumps(config_dict, sort_keys=True)
    hash_full = hashlib.sha256(config_json.encode()).hexdigest()
    return hash_full[:12]


def create_run_folder(
    config: Config,
    timestamp: Optional[str] = None
) -> Path:
    """
    Create a unique folder for a simulation run.

    Folder name format: run_<YYYYmmdd_HHMMSS>_<hash>

    Parameters
    ----------
    config : Config
        Configuration for the run.
    timestamp : str, optional
        Timestamp string. If None, uses current time.

    Returns
    -------
    run_path : Path
        Path to the created run folder.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_hash = compute_config_hash(config)
    folder_name = f"run_{timestamp}_{config_hash}"

    run_path = Path(config.run.out_dir) / folder_name
    run_path.mkdir(parents=True, exist_ok=True)

    return run_path


def save_results(
    result: SweepResult,
    run_path: Union[str, Path]
) -> dict[str, Path]:
    """
    Save sweep results to a run folder.

    Saves:
    - config_resolved.yaml: The resolved configuration
    - results.json: Metadata and arrays as JSON
    - results.csv: Long-format results for analysis

    Parameters
    ----------
    result : SweepResult
        Sweep results to save.
    run_path : str or Path
        Path to the run folder.

    Returns
    -------
    paths : dict
        Dictionary mapping output names to file paths.
    """
    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save resolved config
    config_path = run_path / "config_resolved.yaml"
    save_config(result.config, config_path)
    paths["config"] = config_path

    # Save results as JSON
    json_path = run_path / "results.json"
    json_data = _result_to_json_dict(result)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    paths["json"] = json_path

    # Save results as CSV (long format)
    csv_path = run_path / "results.csv"
    df = _result_to_dataframe(result)
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    return paths


def _result_to_json_dict(result: SweepResult) -> dict:
    """Convert SweepResult to JSON-serializable dictionary."""
    return {
        "metadata": {
            "config_hash": result.config_hash,
            "timestamp": result.timestamp,
            "elapsed_seconds": result.elapsed_seconds,
            "total_points": result.total_points
        },
        "grid": {
            "df_values": result.df_values.tolist(),
            "k_values": result.k_values.tolist(),
            "n_df": len(result.df_values),
            "n_k": len(result.k_values)
        },
        "results": {
            "C_mean": result.C_mean.tolist(),
            "C_min": result.C_min.tolist(),
            "C_max": result.C_max.tolist(),
            "C_std": result.C_std.tolist(),
            "stable_mask": result.stable_mask.tolist(),
            "mode_ids": result.mode_ids.tolist(),
            "phase_slip_counts": result.phase_slip_counts.tolist(),
            "cluster_counts": result.cluster_counts.tolist()
        },
        "config": result.config.to_dict()
    }


def _result_to_dataframe(result: SweepResult) -> pd.DataFrame:
    """Convert SweepResult to long-format DataFrame."""
    rows = []
    for i_k, k_val in enumerate(result.k_values):
        for i_df, df_val in enumerate(result.df_values):
            rows.append({
                "df_over_f0": df_val,
                "k_over_gamma": k_val,
                "C_mean": result.C_mean[i_k, i_df],
                "C_min": result.C_min[i_k, i_df],
                "C_max": result.C_max[i_k, i_df],
                "C_std": result.C_std[i_k, i_df],
                "stable": result.stable_mask[i_k, i_df],
                "mode_id": result.mode_ids[i_k, i_df],
                "mode": mode_to_string(ModeType(result.mode_ids[i_k, i_df])),
                "phase_slip_count": result.phase_slip_counts[i_k, i_df],
                "cluster_count": result.cluster_counts[i_k, i_df]
            })
    return pd.DataFrame(rows)


def load_results(run_path: Union[str, Path]) -> SweepResult:
    """
    Load sweep results from a run folder.

    Parameters
    ----------
    run_path : str or Path
        Path to the run folder.

    Returns
    -------
    result : SweepResult
        Loaded sweep results.
    """
    from .config import load_config

    run_path = Path(run_path)

    # Load config
    config_path = run_path / "config_resolved.yaml"
    config = load_config(config_path)

    # Load JSON results
    json_path = run_path / "results.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # Reconstruct SweepResult
    return SweepResult(
        df_values=np.array(data["grid"]["df_values"]),
        k_values=np.array(data["grid"]["k_values"]),
        C_mean=np.array(data["results"]["C_mean"]),
        C_min=np.array(data["results"]["C_min"]),
        C_max=np.array(data["results"]["C_max"]),
        C_std=np.array(data["results"]["C_std"]),
        stable_mask=np.array(data["results"]["stable_mask"], dtype=np.bool_),
        mode_ids=np.array(data["results"]["mode_ids"], dtype=np.int32),
        phase_slip_counts=np.array(data["results"]["phase_slip_counts"], dtype=np.int32),
        cluster_counts=np.array(data["results"]["cluster_counts"], dtype=np.int32),
        config=config,
        config_hash=data["metadata"]["config_hash"],
        timestamp=data["metadata"]["timestamp"],
        elapsed_seconds=data["metadata"]["elapsed_seconds"],
        total_points=data["metadata"]["total_points"]
    )


def export_results(
    run_path: Union[str, Path],
    formats: list[str] = ["json", "csv"]
) -> dict[str, Path]:
    """
    Export results from a run folder in specified formats.

    This is useful for re-exporting or converting existing results.

    Parameters
    ----------
    run_path : str or Path
        Path to the run folder.
    formats : list of str
        Formats to export ("json", "csv").

    Returns
    -------
    paths : dict
        Dictionary mapping format names to file paths.
    """
    run_path = Path(run_path)
    result = load_results(run_path)

    paths = {}

    if "json" in formats:
        json_path = run_path / "results.json"
        json_data = _result_to_json_dict(result)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        paths["json"] = json_path

    if "csv" in formats:
        csv_path = run_path / "results.csv"
        df = _result_to_dataframe(result)
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path

    return paths


def list_runs(out_dir: Union[str, Path] = "out") -> list[Path]:
    """
    List all run folders in an output directory.

    Parameters
    ----------
    out_dir : str or Path
        Output directory to search.

    Returns
    -------
    runs : list of Path
        List of run folder paths, sorted by name (most recent last).
    """
    out_path = Path(out_dir)
    if not out_path.exists():
        return []

    runs = [p for p in out_path.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs)


def get_latest_run(out_dir: Union[str, Path] = "out") -> Optional[Path]:
    """
    Get the most recent run folder.

    Parameters
    ----------
    out_dir : str or Path
        Output directory to search.

    Returns
    -------
    run_path : Path or None
        Path to most recent run, or None if no runs exist.
    """
    runs = list_runs(out_dir)
    return runs[-1] if runs else None


def verify_reproducibility(
    run_path: Union[str, Path]
) -> bool:
    """
    Verify that a run can be reproduced with the same results.

    Reruns the sweep with the saved config and compares the hash.

    Parameters
    ----------
    run_path : str or Path
        Path to the run folder.

    Returns
    -------
    reproducible : bool
        True if rerun produces identical results.
    """
    from .sweep import run_sweep

    run_path = Path(run_path)
    original = load_results(run_path)

    # Rerun with same config
    new_result = run_sweep(original.config)

    # Compare hashes
    return new_result.config_hash == original.config_hash
