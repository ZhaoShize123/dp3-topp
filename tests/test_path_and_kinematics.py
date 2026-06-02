import numpy as np
import pytest

from dp3_topp.kinematics import path_time_derivatives
from dp3_topp.path_data import PathData


def test_path_data_requires_strictly_increasing_s_and_matching_shapes():
    s = np.array([0.0, 0.5, 1.0])
    q = np.zeros((3, 2))
    path = PathData(s=s, q=q, q_s=q + 1.0, q_ss=q + 2.0, q_sss=q + 3.0)

    assert path.dof == 2
    assert path.samples == 3

    with pytest.raises(ValueError, match="strictly increasing"):
        PathData(s=np.array([0.0, 0.0, 1.0]), q=q, q_s=q, q_ss=q, q_sss=q)

    with pytest.raises(ValueError, match="same shape"):
        PathData(s=s, q=q, q_s=np.zeros((3, 3)), q_ss=q, q_sss=q)


def test_path_data_requires_normalized_unit_interval_domain():
    q = np.zeros((3, 1))

    with pytest.raises(ValueError, match=r"s.*\[0, 1\]"):
        PathData(s=np.array([0.0, 1.0, 2.0]), q=q, q_s=q, q_ss=q, q_sss=q)

    with pytest.raises(ValueError, match=r"s.*\[0, 1\]"):
        PathData(s=np.array([0.1, 0.5, 1.0]), q=q, q_s=q, q_ss=q, q_sss=q)


def test_path_data_rejects_nonfinite_values():
    s = np.array([0.0, 0.5, 1.0])
    q = np.zeros((3, 1))

    with pytest.raises(ValueError, match="finite"):
        PathData(s=np.array([0.0, np.nan, 1.0]), q=q, q_s=q, q_ss=q, q_sss=q)

    bad_q = q.copy()
    bad_q[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        PathData(s=s, q=bad_q, q_s=q, q_ss=q, q_sss=q)


def test_path_data_csv_accepts_utf8_bom_header(tmp_path):
    path = tmp_path / "path.csv"
    path.write_bytes(
        "\ufeffs,q1,dq1,d2q1,d3q1\n0,0,1,0,0\n1,1,1,0,0\n".encode("utf-8")
    )

    data = PathData.from_csv(path)

    assert data.samples == 2
    assert data.dof == 1


def test_path_data_csv_accepts_plan_derivative_column_names(tmp_path):
    path = tmp_path / "path.csv"
    path.write_text(
        "s,q1,q2,dq1_ds,dq2_ds,d2q1_ds2,d2q2_ds2,d3q1_ds3,d3q2_ds3\n"
        "0.0,0.0,0.5,1.0,2.0,0.1,0.2,0.01,0.02\n"
        "1.0,1.0,1.5,1.0,2.0,0.3,0.4,0.03,0.04\n",
        encoding="utf-8",
    )

    data = PathData.from_csv(path)

    assert data.samples == 2
    assert data.dof == 2
    np.testing.assert_allclose(data.q_s, np.array([[1.0, 2.0], [1.0, 2.0]]))
    np.testing.assert_allclose(data.q_ss, np.array([[0.1, 0.2], [0.3, 0.4]]))
    np.testing.assert_allclose(data.q_sss, np.array([[0.01, 0.02], [0.03, 0.04]]))


def test_path_data_csv_rejects_noncontiguous_joint_columns(tmp_path):
    path = tmp_path / "bad_axis_numbering.csv"
    path.write_text(
        "s,q1,q3,dq1,dq2,d2q1,d2q2,d3q1,d3q2\n"
        "0,0,0,1,1,0,0,0,0\n"
        "1,1,1,1,1,0,0,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="q columns must be contiguous"):
        PathData.from_csv(path)


def test_path_time_derivatives_follow_paper_equations():
    q_s = np.array([[3.0, -2.0]])
    q_ss = np.array([[4.0, 1.5]])
    q_sss = np.array([[5.0, -1.0]])

    result = path_time_derivatives(q_s=q_s, q_ss=q_ss, q_sss=q_sss, z=np.array([4.0]), z_s=np.array([2.0]), z_ss=np.array([1.0]))

    assert result.q_dot[0].tolist() == pytest.approx([6.0, -4.0])
    assert result.q_ddot[0].tolist() == pytest.approx([19.0, 4.0])
    assert result.q_jerk[0].tolist() == pytest.approx([67.0, -1.0])


@pytest.mark.parametrize(
    "field",
    ("q_s", "q_ss", "q_sss", "z", "z_s", "z_ss"),
)
def test_path_time_derivatives_reject_nonfinite_inputs(field):
    args = {
        "q_s": np.array([[1.0]]),
        "q_ss": np.array([[0.0]]),
        "q_sss": np.array([[0.0]]),
        "z": np.array([1.0]),
        "z_s": np.array([0.0]),
        "z_ss": np.array([0.0]),
    }
    bad = args[field].copy()
    bad.flat[0] = np.nan
    args[field] = bad

    with pytest.raises(ValueError, match="finite"):
        path_time_derivatives(**args)
