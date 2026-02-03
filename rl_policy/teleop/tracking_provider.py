from __future__ import annotations

import numpy as np


class TrackingZProvider:
    def __init__(self, window_size: int, z_dim: int = 256):
        self.window_size = window_size
        self.z_dim = z_dim
        self._latest: np.ndarray | None = None

    def update(self, z: np.ndarray) -> None:
        self._latest = z.astype(np.float32)

    def latest(self) -> np.ndarray:
        if self._latest is None:
            return np.zeros(self.z_dim, dtype=np.float32)
        return self._latest
