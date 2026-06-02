from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeDerivatives:
    q_dot: np.ndarray
    q_ddot: np.ndarray
    q_jerk: np.ndarray


def path_time_derivatives(
    *,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    q_sss: np.ndarray,
    z: np.ndarray,
    z_s: np.ndarray,
    z_ss: np.ndarray,
) -> TimeDerivatives:
    q_s = np.asarray(q_s, dtype=np.float64)
    q_ss = np.asarray(q_ss, dtype=np.float64)
    q_sss = np.asarray(q_sss, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    z_ss = np.asarray(z_ss, dtype=np.float64)
    if q_s.shape != q_ss.shape or q_s.shape != q_sss.shape:
        raise ValueError("q_s, q_ss, and q_sss must have the same shape")
    if z.ndim != 1 or z_s.shape != z.shape or z_ss.shape != z.shape:
        raise ValueError("z, z_s, and z_ss must be one-dimensional arrays with the same shape")
    if q_s.shape[0] != z.size:
        raise ValueError("path derivative rows must match z samples")
    for name, values in (("q_s", q_s), ("q_ss", q_ss), ("q_sss", q_sss), ("z", z), ("z_s", z_s), ("z_ss", z_ss)):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(z < 0.0):
        raise ValueError("z must be nonnegative")

    sqrt_z = np.sqrt(z)[:, None]
    z_col = z[:, None]
    z_s_col = z_s[:, None]
    z_ss_col = z_ss[:, None]
    q_dot = q_s * sqrt_z
    q_ddot = q_ss * z_col + 0.5 * q_s * z_s_col
    q_jerk = q_sss * z_col * sqrt_z + 1.5 * q_ss * sqrt_z * z_s_col + 0.5 * q_s * sqrt_z * z_ss_col
    return TimeDerivatives(q_dot=q_dot, q_ddot=q_ddot, q_jerk=q_jerk)
