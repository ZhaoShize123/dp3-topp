from pathlib import Path


TORQUE_CONSTRAINED_CHARTS = {
    "joint_velocity": ("Torque-constrained joint velocity", "Joint Velocity"),
    "joint_acceleration": ("Torque-constrained joint acceleration", "Joint Acceleration"),
    "joint_jerk": ("Torque-constrained joint jerk", "Joint Jerk"),
    "joint_torque": ("Torque-constrained joint torque", "Joint Torque"),
    "joint_torque_rate": ("Torque-constrained joint torque rate", "Joint Torque Rate"),
}


def test_homepage_embeds_torque_constrained_long_path_joint_curve_charts():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Torque-constrained DP3 joint-data plots" in readme
    assert "long_path_01_torque_speed_drop" in readme
    assert "velocity-dependent torque constraint" in readme
    assert "assets/simple-story/" not in readme

    for plot_name, (alt_text, svg_title) in TORQUE_CONSTRAINED_CHARTS.items():
        filename = f"long_path_01_torque_speed_drop_{plot_name}.svg"
        assert f"![{alt_text}](assets/complex-constraint-sweep/{filename})" in readme

        svg_path = root / "assets" / "complex-constraint-sweep" / filename
        assert svg_path.exists()
        svg = svg_path.read_text(encoding="utf-8")
        assert svg_title in svg
        assert "<polyline" in svg
        for axis in range(1, 7):
            assert f"joint {axis}" in svg


def test_homepage_uses_torque_limit_comparison_for_torque_curve():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Torque constraint utilization comparison" in readme
    assert "![Torque constraint comparison](assets/joint-torque-normalized.svg)" in readme
    for filename in (
        "joint-velocity-normalized.svg",
        "joint-acceleration-normalized.svg",
        "joint-jerk-normalized.svg",
    ):
        assert f"assets/{filename}" not in readme

    torque_svg = (root / "assets" / "joint-torque-normalized.svg").read_text(encoding="utf-8")
    assert "torque constraint comparison" in torque_svg
    assert "velocity-dependent torque constraint" in torque_svg
    assert "tau / torque constraint" in torque_svg
    assert "upper torque constraint" in torque_svg
    assert "lower torque constraint" in torque_svg
    assert "normalized path coordinate s" in torque_svg
    assert "normalized torque utilization" in torque_svg
    for axis in range(1, 7):
        assert f"J{axis}" in torque_svg
        assert f"DP3 J{axis}" in torque_svg
        assert f"TOPPRA J{axis}" in torque_svg
        assert f"+1 torque constraint J{axis}" in torque_svg
        assert f"-1 torque constraint J{axis}" in torque_svg
