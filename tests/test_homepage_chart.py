from pathlib import Path


CHARTS = {
    "joint_velocity": ("Joint velocity", "q_dot", "joint-velocity-normalized.svg"),
    "joint_acceleration": ("Joint acceleration", "q_ddot", "joint-acceleration-normalized.svg"),
    "joint_jerk": ("Joint jerk", "q_jerk", "joint-jerk-normalized.svg"),
    "joint_torque": ("Joint torque", "tau", "joint-torque-normalized.svg"),
}


def test_homepage_embeds_normalized_long_path_joint_charts():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Normalized long-path joint comparisons" in readme
    for title, quantity, filename in CHARTS.values():
        assert f"![{title} comparison](assets/{filename})" in readme
        asset_path = root / "assets" / filename
        assert asset_path.exists()

        svg = asset_path.read_text(encoding="utf-8")
        assert "long_path_01" in svg
        assert "8.61x longer than path_01" in svg
        assert "normalized utilization" in svg
        assert quantity in svg
        assert "DP3" in svg
        assert "TOPPRA" in svg
        for axis in range(1, 7):
            assert f"J{axis}" in svg
