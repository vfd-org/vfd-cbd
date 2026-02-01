"""
Coupling matrix construction for oscillator networks.

Provides functions to create coupling matrices K_ij for different
coupling schemes: ring adjacency and distance decay.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from .geometry import create_geometry, compute_distances, get_ring_neighbors


def create_coupling_matrix(
    N: int,
    k_strength: float,
    geometry: Literal["ring", "circle_decay"],
    neighbor_k: int = 2,
    radius: float = 1.0,
    lambda_decay: float = 0.5,
    cutoff: float = 3.0
) -> NDArray[np.float64]:
    """
    Create the coupling matrix K_ij for a given geometry and parameters.

    Parameters
    ----------
    N : int
        Number of oscillators.
    k_strength : float
        Base coupling strength.
    geometry : {"ring", "circle_decay"}
        Type of coupling geometry.
    neighbor_k : int, optional
        Number of neighbors on each side for ring geometry (default 2).
    radius : float, optional
        Radius for circle_decay geometry (default 1.0).
    lambda_decay : float, optional
        Decay length for distance-decay coupling (default 0.5).
    cutoff : float, optional
        Cutoff distance in units of lambda_decay (default 3.0).

    Returns
    -------
    K : ndarray of shape (N, N)
        Coupling matrix where K[i, j] is the coupling strength from j to i.
    """
    if geometry == "ring":
        return ring_adjacency(N, k_strength, neighbor_k)
    elif geometry == "circle_decay":
        return distance_decay(N, k_strength, radius, lambda_decay, cutoff)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def ring_adjacency(
    N: int,
    k_strength: float,
    neighbor_k: int = 2
) -> NDArray[np.float64]:
    """
    Create a ring adjacency coupling matrix.

    Each node is coupled to its k nearest neighbors on each side.
    The coupling is uniform for all connected pairs.

    Parameters
    ----------
    N : int
        Number of oscillators.
    k_strength : float
        Coupling strength for connected pairs.
    neighbor_k : int, optional
        Number of neighbors on each side (default 2).

    Returns
    -------
    K : ndarray of shape (N, N)
        Sparse coupling matrix (dense array with zeros for non-neighbors).
    """
    K = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        neighbors = get_ring_neighbors(i, N, neighbor_k)
        for j in neighbors:
            K[i, j] = k_strength

    return K


def distance_decay(
    N: int,
    k_strength: float,
    radius: float = 1.0,
    lambda_decay: float = 0.5,
    cutoff: float = 3.0
) -> NDArray[np.float64]:
    """
    Create a distance-decay coupling matrix.

    Nodes are placed on a circle and coupling decays exponentially
    with distance. A cutoff is applied to sparsify the matrix.

    K_ij = k_strength * exp(-r_ij / lambda_decay) if r_ij < cutoff * lambda_decay
         = 0 otherwise

    Parameters
    ----------
    N : int
        Number of oscillators.
    k_strength : float
        Base coupling strength at zero distance.
    radius : float, optional
        Radius of the circle for node placement (default 1.0).
    lambda_decay : float, optional
        Decay length scale (default 0.5).
    cutoff : float, optional
        Cutoff in units of lambda_decay (default 3.0).

    Returns
    -------
    K : ndarray of shape (N, N)
        Coupling matrix with distance-decay values.
    """
    positions = create_geometry(N, "circle_decay", radius)
    distances = compute_distances(positions)

    # Compute exponential decay
    K = k_strength * np.exp(-distances / lambda_decay)

    # Apply cutoff
    cutoff_distance = cutoff * lambda_decay
    K[distances > cutoff_distance] = 0.0

    # Zero out diagonal (no self-coupling)
    np.fill_diagonal(K, 0.0)

    return K


def normalize_coupling(K: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """
    Normalize coupling matrix by loss parameter.

    Returns K / gamma, which gives coupling in units of loss.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Coupling matrix.
    gamma : float
        Loss parameter.

    Returns
    -------
    K_normalized : ndarray of shape (N, N)
        Normalized coupling matrix.
    """
    return K / gamma


def get_effective_coupling(
    K: NDArray[np.float64],
    gamma: float
) -> float:
    """
    Get effective coupling strength (mean of non-zero couplings normalized by gamma).

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Coupling matrix.
    gamma : float
        Loss parameter.

    Returns
    -------
    k_eff : float
        Effective coupling-to-loss ratio.
    """
    nonzero = K[K > 0]
    if len(nonzero) == 0:
        return 0.0
    return np.mean(nonzero) / gamma
