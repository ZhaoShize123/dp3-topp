import re
from pathlib import Path


NORMALIZED_CHARTS = {
    "joint-velocity-normalized.svg": ("Joint velocity limit utilization comparison", "q_dot", "q_dot / q_dot limit"),
    "joint-acceleration-normalized.svg": (
        "Joint acceleration limit utilization comparison",
        "q_ddot",
        "q_ddot / q_ddot limit",
    ),
    "joint-jerk-normalized.svg": ("Joint jerk limit utilization comparison", "q_jerk", "q_jerk / q_jerk limit"),
    "joint-torque-normalized.svg": ("Joint torque limit utilization comparison", "tau", "tau / torque constraint"),
}


def test_homepage_embeds_normalized_limit_utilization_joint_curve_charts():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Normalized limit-utilization joint comparisons" in readme
    assert "Y axis is each joint quantity divided by its corresponding limit" in readme
    assert "TOPPRA is drawn with dashed curves" in readme
    assert "upper/lower limit lines are shown at +1 and -1" in readme
    assert "assets/simple-story/" not in readme

    for filename, (alt_text, _, _) in NORMALIZED_CHARTS.items():
        assert f"![{alt_text}](assets/{filename})" in readme


def test_homepage_normalized_joint_svgs_show_toppra_dashed_and_limit_lines():
    root = Path(__file__).resolve().parents[1]

    for filename, (_, quantity, y_axis_label) in NORMALIZED_CHARTS.items():
        svg = (root / "assets" / filename).read_text(encoding="utf-8")
        assert y_axis_label in svg
        assert "normalized path coordinate s" in svg
        assert "upper/lower limit: +1/-1" in svg
        assert re.search(r'data-series="TOPPRA J1 [^"]*".*stroke-dasharray="6 5"', svg)
        assert re.search(r'data-series="DP3 J1 [^"]*".*stroke-width="2', svg)
        assert re.search(r'data-series="\+1 .* upper .*limit"', svg)
        assert re.search(r'data-series="-1 .* lower .*limit"', svg)

        if quantity != "tau":
            assert f"upper {quantity} limit" in svg
            assert f"lower {quantity} limit" in svg
        else:
            assert "velocity-dependent torque constraint" in svg
            assert "upper torque constraint" in svg
            assert "lower torque constraint" in svg

        for axis in range(1, 7):
            assert f"DP3 J{axis}" in svg
            assert f"TOPPRA J{axis}" in svg
