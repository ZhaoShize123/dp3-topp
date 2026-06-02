import math

import numpy as np
import pytest

from dp3_topp.interpolation import C2LinearZ, C3QuadraticSpeed, C4CubicSpeed


def test_c2_linear_z_matches_endpoint_values_and_derivatives():
    profile = C2LinearZ.from_interval(0.0, 2.0, z0=0.0, z1=8.0)

    left = profile.evaluate(np.array([0.0]))
    right = profile.evaluate(np.array([2.0]))
    middle = profile.evaluate(np.array([1.0]))

    assert left.z[0] == pytest.approx(0.0)
    assert right.z[0] == pytest.approx(8.0)
    assert middle.z[0] == pytest.approx(4.0)
    assert middle.z_s[0] == pytest.approx(4.0)
    assert middle.z_ss[0] == pytest.approx(0.0)
    assert math.isfinite(profile.duration())


def test_c3_quadratic_speed_matches_end_slope_and_rejects_zero_speed():
    profile = C3QuadraticSpeed.from_interval(1.0, 3.0, z0=1.0, z1=4.0, z_s1=2.0)

    values = profile.evaluate(np.array([1.0, 3.0]))

    assert values.z[0] == pytest.approx(1.0)
    assert values.z[1] == pytest.approx(4.0)
    assert values.z_s[1] == pytest.approx(2.0)
    assert np.all(values.s_dot > 0.0)
    assert math.isfinite(profile.duration())

    with pytest.raises(ValueError, match="positive speed"):
        C3QuadraticSpeed.from_interval(0.0, 1.0, z0=0.0, z1=1.0, z_s1=0.0)


def test_c4_cubic_speed_matches_both_endpoint_slopes():
    profile = C4CubicSpeed.from_interval(0.0, 0.5, z0=1.0, z1=2.25, z_s0=0.4, z_s1=-0.2)

    values = profile.evaluate(np.array([0.0, 0.5]))

    assert values.z[0] == pytest.approx(1.0)
    assert values.z[1] == pytest.approx(2.25)
    assert values.z_s[0] == pytest.approx(0.4)
    assert values.z_s[1] == pytest.approx(-0.2)
    assert np.all(values.s_dot > 0.0)


def test_c4_cubic_speed_rejects_nonpositive_internal_speed():
    with pytest.raises(ValueError, match="positive speed"):
        C4CubicSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s0=-20.0, z_s1=20.0)


@pytest.mark.parametrize(
    ("z0", "z1"),
    (
        (np.nan, 1.0),
        (0.0, np.inf),
    ),
)
def test_c2_linear_z_rejects_nonfinite_endpoints(z0, z1):
    with pytest.raises(ValueError, match="finite"):
        C2LinearZ.from_interval(0.0, 1.0, z0=z0, z1=z1)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: C3QuadraticSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s1=np.nan),
        lambda: C4CubicSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s0=np.inf, z_s1=0.0),
        lambda: C4CubicSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s0=0.0, z_s1=np.nan),
    ),
)
def test_polynomial_speed_profiles_reject_nonfinite_endpoint_slopes(factory):
    with pytest.raises(ValueError, match="finite"):
        factory()


def test_profile_evaluation_rejects_nonfinite_sample_points():
    profile = C2LinearZ.from_interval(0.0, 1.0, z0=0.0, z1=1.0)

    with pytest.raises(ValueError, match="finite"):
        profile.evaluate(np.array([np.nan]))


@pytest.mark.parametrize(
    "profile",
    (
        C2LinearZ.from_interval(0.0, 1.0, z0=0.0, z1=1.0),
        C3QuadraticSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s1=0.0),
        C4CubicSpeed.from_interval(0.0, 1.0, z0=1.0, z1=1.0, z_s0=0.0, z_s1=0.0),
    ),
)
def test_profile_evaluation_rejects_samples_outside_interval(profile):
    with pytest.raises(ValueError, match="within the interpolation interval"):
        profile.evaluate(np.array([-1e-6, 0.5]))

    with pytest.raises(ValueError, match="within the interpolation interval"):
        profile.evaluate(np.array([0.5, 1.000001]))
