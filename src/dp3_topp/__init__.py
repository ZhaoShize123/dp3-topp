"""DP3/DP2 time-optimal path-parameterization library."""

from dp3_topp.api import DPToppRunResult, run_dp2, run_dp3
from dp3_topp.constraints import ConstraintAudit, ConstraintLimits, ConstraintViolation, audit_constraints
from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
from dp3_topp.optimizer import (
    DP3Config,
    TrajectoryQuantities,
    TrajectoryResult,
    evaluate_trajectory_quantities,
    optimize_dp2,
    optimize_dp3,
    resample_trajectory_by_segments,
    resample_trajectory_by_time,
)
from dp3_topp.path_data import PathData

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConstraintAudit",
    "ConstraintLimits",
    "ConstraintViolation",
    "DP3Config",
    "DPToppRunResult",
    "MujocoRobotDynamics",
    "PathData",
    "TrajectoryQuantities",
    "TrajectoryResult",
    "audit_constraints",
    "evaluate_trajectory_quantities",
    "optimize_dp2",
    "optimize_dp3",
    "resample_trajectory_by_segments",
    "resample_trajectory_by_time",
    "run_dp2",
    "run_dp3",
]
