from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_POSITIVE_SPEED_TOL = 1e-12


@dataclass(frozen=True)
class ProfileValues:
    s: np.ndarray
    z: np.ndarray
    z_s: np.ndarray
    z_ss: np.ndarray
    s_dot: np.ndarray


def _as_points(s: np.ndarray | float) -> np.ndarray:
    points = np.atleast_1d(np.asarray(s, dtype=np.float64))
    if np.any(~np.isfinite(points)):
        raise ValueError("profile sample points must be finite")
    return points


def _check_points_in_interval(points: np.ndarray, s0: float, s1: float) -> np.ndarray:
    if np.any(points < float(s0) - 1e-12) or np.any(points > float(s1) + 1e-12):
        raise ValueError("profile sample points must lie within the interpolation interval")
    return np.clip(points, float(s0), float(s1))


def _check_interval(s0: float, s1: float) -> float:
    h = float(s1) - float(s0)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Interpolation interval must have positive length")
    return h


def _positive_root(z: float) -> float:
    z = float(z)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("C3/C4 profiles require positive speed endpoints")
    return float(np.sqrt(z))


def _duration_by_gauss(h: float, speed_poly) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(64)
    xi = 0.5 * (nodes - 1.0)
    w = 0.5 * weights
    speed = speed_poly(xi)
    if np.any(speed <= _POSITIVE_SPEED_TOL):
        raise ValueError("C3/C4 profiles require positive speed over the full interval")
    return float(h * np.sum(w / speed))


@dataclass(frozen=True)
class C2LinearZ:
    s0: float
    s1: float
    z0: float
    z1: float

    @classmethod
    def from_interval(cls, s0: float, s1: float, z0: float, z1: float) -> "C2LinearZ":
        _check_interval(s0, s1)
        z0 = float(z0)
        z1 = float(z1)
        if not np.isfinite(z0) or not np.isfinite(z1):
            raise ValueError("C2 profile endpoints must be finite")
        if min(z0, z1) < 0.0:
            raise ValueError("C2 profile requires nonnegative z endpoints")
        return cls(float(s0), float(s1), z0, z1)

    @property
    def h(self) -> float:
        return self.s1 - self.s0

    def _xi(self, s: np.ndarray) -> np.ndarray:
        return (s - self.s1) / self.h

    def evaluate(self, s: np.ndarray | float) -> ProfileValues:
        points = _check_points_in_interval(_as_points(s), self.s0, self.s1)
        xi = self._xi(points)
        c0 = self.z1
        c1 = self.z1 - self.z0
        z = c1 * xi + c0
        z = np.maximum(z, 0.0)
        z_s = np.full_like(z, c1 / self.h)
        z_ss = np.zeros_like(z)
        return ProfileValues(s=points, z=z, z_s=z_s, z_ss=z_ss, s_dot=np.sqrt(z))

    def duration(self) -> float:
        c0 = self.z1
        c1 = self.z1 - self.z0
        if abs(c1) <= 1e-14:
            if c0 <= 0.0:
                raise ValueError("C2 zero-speed segment has infinite duration")
            return float(self.h / np.sqrt(c0))
        lo = np.sqrt(max(c0 - c1, 0.0))
        hi = np.sqrt(max(c0, 0.0))
        return float(self.h * 2.0 * (hi - lo) / c1)


@dataclass(frozen=True)
class _PolynomialSpeedProfile:
    s0: float
    s1: float
    coefficients: tuple[float, ...]

    @property
    def h(self) -> float:
        return self.s1 - self.s0

    def _xi(self, s: np.ndarray) -> np.ndarray:
        return (s - self.s1) / self.h

    def _poly(self, xi: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(xi, self.coefficients)

    def _poly_d1(self, xi: np.ndarray) -> np.ndarray:
        coeff = np.polynomial.polynomial.polyder(self.coefficients, 1)
        return np.polynomial.polynomial.polyval(xi, coeff)

    def _poly_d2(self, xi: np.ndarray) -> np.ndarray:
        coeff = np.polynomial.polynomial.polyder(self.coefficients, 2)
        return np.polynomial.polynomial.polyval(xi, coeff)

    def _validate_positive_speed(self) -> None:
        grid = np.linspace(-1.0, 0.0, 257)
        candidates = [grid]
        d1 = np.polynomial.polynomial.polyder(self.coefficients, 1)
        if len(d1) > 1:
            roots = np.roots(np.asarray(d1[::-1], dtype=np.float64))
            real_roots = roots[np.isclose(roots.imag, 0.0)].real
            real_roots = real_roots[(real_roots >= -1.0) & (real_roots <= 0.0)]
            if real_roots.size:
                candidates.append(real_roots)
        xi = np.concatenate(candidates)
        if np.any(self._poly(xi) <= _POSITIVE_SPEED_TOL):
            raise ValueError("C3/C4 profiles require positive speed over the full interval")

    def evaluate(self, s: np.ndarray | float) -> ProfileValues:
        points = _check_points_in_interval(_as_points(s), self.s0, self.s1)
        xi = self._xi(points)
        p = self._poly(xi)
        if np.any(p <= _POSITIVE_SPEED_TOL):
            raise ValueError("C3/C4 profiles require positive speed at evaluation points")
        p1 = self._poly_d1(xi)
        p2 = self._poly_d2(xi)
        z = p * p
        z_xi = 2.0 * p * p1
        z_xixi = 2.0 * (p1 * p1 + p * p2)
        return ProfileValues(
            s=points,
            z=z,
            z_s=z_xi / self.h,
            z_ss=z_xixi / (self.h * self.h),
            s_dot=p,
        )

    def duration(self) -> float:
        return _duration_by_gauss(self.h, self._poly)


@dataclass(frozen=True)
class C3QuadraticSpeed(_PolynomialSpeedProfile):
    @classmethod
    def from_interval(cls, s0: float, s1: float, z0: float, z1: float, z_s1: float) -> "C3QuadraticSpeed":
        h = _check_interval(s0, s1)
        r0 = _positive_root(z0)
        r1 = _positive_root(z1)
        z_s1 = float(z_s1)
        if not np.isfinite(z_s1):
            raise ValueError("C3 endpoint slope must be finite")
        z_xi_1 = z_s1 * h
        c0 = r1
        c1 = z_xi_1 / (2.0 * r1)
        c2 = r0 + c1 - c0
        profile = cls(float(s0), float(s1), (c0, c1, c2))
        profile._validate_positive_speed()
        return profile


@dataclass(frozen=True)
class C4CubicSpeed(_PolynomialSpeedProfile):
    @classmethod
    def from_interval(
        cls,
        s0: float,
        s1: float,
        z0: float,
        z1: float,
        z_s0: float,
        z_s1: float,
    ) -> "C4CubicSpeed":
        h = _check_interval(s0, s1)
        r0 = _positive_root(z0)
        r1 = _positive_root(z1)
        z_s0 = float(z_s0)
        z_s1 = float(z_s1)
        if not np.isfinite(z_s0) or not np.isfinite(z_s1):
            raise ValueError("C4 endpoint slopes must be finite")
        z_xi_0 = z_s0 * h
        z_xi_1 = z_s1 * h
        d = r1
        c = z_xi_1 / (2.0 * r1)
        endpoint_delta = r0 + c - d
        left_slope = z_xi_0 / (2.0 * r0)
        a = left_slope - c + 2.0 * endpoint_delta
        b = endpoint_delta + a
        profile = cls(float(s0), float(s1), (d, c, b, a))
        profile._validate_positive_speed()
        return profile
