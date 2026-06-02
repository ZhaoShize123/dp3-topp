from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PathData:
    s: np.ndarray
    q: np.ndarray
    q_s: np.ndarray
    q_ss: np.ndarray
    q_sss: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=np.float64)
        q = np.asarray(self.q, dtype=np.float64)
        q_s = np.asarray(self.q_s, dtype=np.float64)
        q_ss = np.asarray(self.q_ss, dtype=np.float64)
        q_sss = np.asarray(self.q_sss, dtype=np.float64)

        if s.ndim != 1 or s.size < 2:
            raise ValueError("s must be a one-dimensional array with at least two samples")
        if np.any(~np.isfinite(s)):
            raise ValueError("s must contain only finite values")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("s must be strictly increasing")
        if abs(float(s[0])) > 1e-12 or abs(float(s[-1]) - 1.0) > 1e-12:
            raise ValueError("s must span the normalized path domain [0, 1]")
        if q.ndim != 2 or q.shape[0] != s.size:
            raise ValueError("q must be a two-dimensional array with one row per s sample")
        if q_s.shape != q.shape or q_ss.shape != q.shape or q_sss.shape != q.shape:
            raise ValueError("q, q_s, q_ss, and q_sss must have the same shape")
        for name, values in (("q", q), ("q_s", q_s), ("q_ss", q_ss), ("q_sss", q_sss)):
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")

        object.__setattr__(self, "s", s)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "q_s", q_s)
        object.__setattr__(self, "q_ss", q_ss)
        object.__setattr__(self, "q_sss", q_sss)

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])

    @property
    def samples(self) -> int:
        return int(self.s.size)

    @classmethod
    def from_csv(cls, path: Path) -> "PathData":
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
        if data.size == 0 or data.dtype.names is None:
            raise ValueError(f"Empty or headerless path CSV: {path}")
        original_names = list(data.dtype.names)
        name_map = {name.lstrip("\ufeff"): name for name in original_names}
        names = list(name_map)
        if "s" not in names:
            raise ValueError("path CSV must contain an s column")
        q_names = [name for name in names if name.startswith("q") and name[1:].isdigit()]
        if not q_names:
            raise ValueError("path CSV must contain q1..qn columns")
        q_names.sort(key=lambda item: int(item[1:]))
        q_axes = [int(name[1:]) for name in q_names]
        expected_axes = list(range(1, len(q_names) + 1))
        if q_axes != expected_axes:
            raise ValueError(f"q columns must be contiguous q1..q{len(q_names)}")
        dof = len(q_names)

        def columns(candidates_by_axis: list[list[str]]) -> np.ndarray:
            cols = []
            missing = []
            for candidates in candidates_by_axis:
                found = next((name for name in candidates if name in names), None)
                if found is None:
                    missing.append(candidates[0])
                else:
                    cols.append(found)
            if missing:
                raise ValueError(f"path CSV missing columns: {', '.join(missing)}")
            return np.column_stack([np.asarray(data[name_map[name]], dtype=np.float64) for name in cols])

        def derivative_columns(prefix: str, suffix: str) -> list[list[str]]:
            return [[f"{prefix}{idx}{suffix}", f"{prefix}{idx}"] for idx in range(1, dof + 1)]

        return cls(
            s=np.asarray(data[name_map["s"]], dtype=np.float64),
            q=columns([[f"q{idx}"] for idx in range(1, dof + 1)]),
            q_s=columns(derivative_columns("dq", "_ds")),
            q_ss=columns(derivative_columns("d2q", "_ds2")),
            q_sss=columns(derivative_columns("d3q", "_ds3")),
        )
