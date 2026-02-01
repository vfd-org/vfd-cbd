"""
Geometry definitions for oscillator networks.

Provides functions to create node positions for different geometries.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal


def create_geometry(
    N: int,
    geometry: Literal["ring", "circle_decay"],
    radius: float = 1.0
) -> NDArray[np.float64]:
    """
    Create node positions for a given geometry.

    Parameters
    ----------
    N : int
        Number of oscillators.
    geometry : {"ring", "circle_decay"}
        Type of geometry.
    radius : float, optional
        Radius for circular arrangements (default 1.0).

    Returns
    -------
    positions : ndarray of shape (N, 2)
        2D positions of each oscillator.
    """
    if geometry == "ring":
        return ring_positions(N)
    elif geometry == "circle_decay":
        return circle_positions(N, radius)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def ring_positions(N: int) -> NDArray[np.float64]:
    """
    Create positions for a ring topology.

    Nodes are placed on a unit circle, evenly spaced.
    This is used for ring adjacency coupling.

    Parameters
    ----------
    N : int
        Number of oscillators.

    Returns
    -------
    positions : ndarray of shape (N, 2)
        2D positions of each oscillator on unit circle.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    return positions


def circle_positions(N: int, radius: float = 1.0) -> NDArray[np.float64]:
    """
    Create positions for a circle with specified radius.

    Nodes are placed on a circle of given radius, evenly spaced.
    This is used for distance-decay coupling.

    Parameters
    ----------
    N : int
        Number of oscillators.
    radius : float, optional
        Radius of the circle (default 1.0).

    Returns
    -------
    positions : ndarray of shape (N, 2)
        2D positions of each oscillator.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return positions


def compute_distances(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute pairwise distances between all nodes.

    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        2D positions of each oscillator.

    Returns
    -------
    distances : ndarray of shape (N, N)
        Pairwise distance matrix where distances[i, j] is the
        Euclidean distance between nodes i and j.
    """
    N = positions.shape[0]
    # Vectorized distance computation
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


def get_ring_neighbors(i: int, N: int, k: int) -> list[int]:
    """
    Get the k-nearest neighbors on each side for node i in a ring.

    Parameters
    ----------
    i : int
        Index of the node.
    N : int
        Total number of nodes.
    k : int
        Number of neighbors on each side.

    Returns
    -------
    neighbors : list of int
        Indices of the 2k neighbors (k on each side), with wraparound.
    """
    neighbors = []
    for offset in range(1, k + 1):
        neighbors.append((i - offset) % N)
        neighbors.append((i + offset) % N)
    return neighbors
