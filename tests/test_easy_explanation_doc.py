from pathlib import Path


def test_easy_explanation_doc_uses_simple_story_images():
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "dp3_easy_explanation_zh.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    required_terms = [
        "DP3 算法实现过程：简单图文版",
        "路线册",
        "z = s_dot^2",
        "过河踩石头",
        "守门人",
        "力矩约束",
        "jerk 约束",
        "MuJoCo 动力学",
        "trajectory.csv",
        "quantities.csv",
        "../assets/simple-story/01-road-map.svg",
        "../assets/simple-story/02-stepping-stones.svg",
        "../assets/simple-story/03-constraint-gates.svg",
        "../assets/simple-story/04-ledger-charts.svg",
    ]
    for term in required_terms:
        assert term in doc

    assert "docs/dp3_easy_explanation_zh.md" in readme


def test_easy_explanation_assets_are_static_images_not_html():
    root = Path(__file__).resolve().parents[1]
    asset_dir = root / "assets" / "simple-story"
    expected = [
        "01-road-map.svg",
        "02-stepping-stones.svg",
        "03-constraint-gates.svg",
        "04-ledger-charts.svg",
    ]
    for filename in expected:
        path = asset_dir / filename
        assert path.exists(), filename
        svg = path.read_text(encoding="utf-8")
        assert svg.startswith("<svg")
        assert "<html" not in svg.lower()
        assert "<rect" in svg
        assert "<path" in svg
