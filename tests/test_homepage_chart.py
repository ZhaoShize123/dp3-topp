from pathlib import Path


def test_homepage_embeds_joint_data_comparison_chart():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    asset_path = root / "assets" / "dp3-vs-toppra-joint-jerk.svg"

    assert "![Joint jerk comparison](assets/dp3-vs-toppra-joint-jerk.svg)" in readme
    assert asset_path.exists()

    svg = asset_path.read_text(encoding="utf-8")
    assert "Joint 1 jerk utilization" in svg
    assert "q_jerk1" in svg
    assert "DP3 peak: 0.98x limit" in svg
    assert "TOPPRA peak: 77.18x limit" in svg
    assert "third-order joint constraint" in svg
