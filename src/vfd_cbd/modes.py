"""
Failure mode classification for coupled oscillator systems.

Provides deterministic classification of system behavior into
distinct failure modes based on coherence statistics.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

from .metrics import CoherenceStats


class ModeType(IntEnum):
    """
    Enumeration of failure mode types.

    Integer values are used for array storage and plotting.
    """
    STABLE = 0
    DECOHERENT = 1
    PHASE_SLIP = 2
    CLUSTER_SPLIT = 3
    INVERSION = 4
    UNKNOWN = 5


# Mode names for display
MODE_NAMES = {
    ModeType.STABLE: "stable",
    ModeType.DECOHERENT: "decoherent",
    ModeType.PHASE_SLIP: "phase_slip",
    ModeType.CLUSTER_SPLIT: "cluster_split",
    ModeType.INVERSION: "inversion",
    ModeType.UNKNOWN: "unknown",
}

# Mode colors for plotting
MODE_COLORS = {
    ModeType.STABLE: "#2ecc71",       # Green
    ModeType.DECOHERENT: "#e74c3c",   # Red
    ModeType.PHASE_SLIP: "#f39c12",   # Orange
    ModeType.CLUSTER_SPLIT: "#9b59b6", # Purple
    ModeType.INVERSION: "#3498db",    # Blue
    ModeType.UNKNOWN: "#95a5a6",      # Gray
}


@dataclass
class ModeStats:
    """Statistics used for mode classification."""
    stats: CoherenceStats
    phase_slip_count: int
    cluster_count: int
    inversion_flag: bool


def classify_mode(
    stats: CoherenceStats,
    phase_slip_count: int,
    cluster_count: int,
    inversion_flag: bool,
    C_threshold: float = 0.7,
    slip_threshold: int = 5,
    cluster_threshold: int = 2,
    cluster_C_low: float = 0.3,
    cluster_C_high: float = 0.8
) -> ModeType:
    """
    Classify the failure mode based on simulation statistics.

    Classification logic (in order of priority):
    1. phase_slip: phase_slip_count >= slip_threshold
    2. cluster_split: cluster_count >= cluster_threshold AND C_mean in [cluster_C_low, cluster_C_high]
    3. inversion: inversion_flag is True
    4. stable: C_mean >= C_threshold
    5. decoherent: otherwise

    Parameters
    ----------
    stats : CoherenceStats
        Coherence statistics from simulation.
    phase_slip_count : int
        Number of detected phase slip events.
    cluster_count : int
        Number of detected phase clusters.
    inversion_flag : bool
        Whether phase inversion was detected.
    C_threshold : float, optional
        Threshold for stability (default 0.7).
    slip_threshold : int, optional
        Minimum slips for phase_slip mode (default 5).
    cluster_threshold : int, optional
        Minimum clusters for cluster_split mode (default 2).
    cluster_C_low : float, optional
        Lower bound of C_mean for cluster_split detection (default 0.3).
    cluster_C_high : float, optional
        Upper bound of C_mean for cluster_split detection (default 0.8).

    Returns
    -------
    mode : ModeType
        Classified failure mode.
    """
    C_mean = stats.C_mean

    # Priority 1: Phase slip detection
    if phase_slip_count >= slip_threshold:
        return ModeType.PHASE_SLIP

    # Priority 2: Cluster split detection
    if cluster_count >= cluster_threshold:
        if cluster_C_low <= C_mean <= cluster_C_high:
            return ModeType.CLUSTER_SPLIT

    # Priority 3: Inversion detection
    if inversion_flag:
        return ModeType.INVERSION

    # Priority 4: Stability check
    if C_mean >= C_threshold:
        return ModeType.STABLE

    # Default: Decoherent
    return ModeType.DECOHERENT


def mode_to_string(mode: ModeType) -> str:
    """
    Convert mode type to human-readable string.

    Parameters
    ----------
    mode : ModeType
        Mode type to convert.

    Returns
    -------
    name : str
        Human-readable mode name.
    """
    return MODE_NAMES.get(mode, "unknown")


def string_to_mode(name: str) -> ModeType:
    """
    Convert string to mode type.

    Parameters
    ----------
    name : str
        Mode name string.

    Returns
    -------
    mode : ModeType
        Corresponding mode type.
    """
    name_lower = name.lower()
    for mode, mode_name in MODE_NAMES.items():
        if mode_name == name_lower:
            return mode
    return ModeType.UNKNOWN


def get_mode_color(mode: ModeType) -> str:
    """
    Get the color for a mode type (for plotting).

    Parameters
    ----------
    mode : ModeType
        Mode type.

    Returns
    -------
    color : str
        Hex color string.
    """
    return MODE_COLORS.get(mode, "#95a5a6")


def describe_mode(mode: ModeType) -> str:
    """
    Get a description of what the mode means.

    Parameters
    ----------
    mode : ModeType
        Mode type to describe.

    Returns
    -------
    description : str
        Human-readable description.
    """
    descriptions = {
        ModeType.STABLE: "System maintains coherent oscillation with high synchronization.",
        ModeType.DECOHERENT: "System loses coherence; oscillators become desynchronized.",
        ModeType.PHASE_SLIP: "System experiences phase slip events (2π jumps in relative phase).",
        ModeType.CLUSTER_SPLIT: "Oscillators split into multiple synchronized clusters.",
        ModeType.INVERSION: "Mean phase is inverted relative to the drive phase.",
        ModeType.UNKNOWN: "System state could not be classified.",
    }
    return descriptions.get(mode, "Unknown mode.")
