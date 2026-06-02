from pathlib import Path


def test_homepage_embeds_dp3_dp2_comparison_chart():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    asset_path = root / "assets" / "dp3-vs-dp2-comparison.svg"

    assert "![DP3 vs DP2 comparison](assets/dp3-vs-dp2-comparison.svg)" in readme
    assert asset_path.exists()

    svg = asset_path.read_text(encoding="utf-8")
    assert "14 T12A paths" in svg
    assert "DP3 total: 194.35 s" in svg
    assert "DP2 total: 196.00 s" in svg
    assert "1.65 s faster" in svg
    assert "0 violations" in svg
