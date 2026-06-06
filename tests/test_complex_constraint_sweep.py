from pathlib import Path


def test_complex_constraint_sweep_script_documents_torque_and_jerk_cases():
    root = Path(__file__).resolve().parents[1]
    script = (root / "examples" / "complex_constraint_sweep.py").read_text(encoding="utf-8")
    doc = (root / "docs" / "dp3_complex_constraint_sweep_zh.md").read_text(encoding="utf-8")

    for term in (
        "torque_speed_drop",
        "tight_jerk",
        "combined_torque_jerk",
        "velocity_dependent_torque",
        "joint_torque.svg",
        "joint_jerk.svg",
        "gallery.html",
    ):
        assert term in script

    assert "力矩与 jerk 约束测试图集" in doc
    assert "velocity-dependent torque constraint" in doc
    assert "outputs/runs/dp3-complex-constraint-sweep/gallery.html" in doc

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Complex torque and jerk constraint sweep" in readme
    assert "docs/dp3_complex_constraint_sweep_zh.md" in readme
    assert "assets/complex-constraint-sweep/" in readme
    assert "assets/simple-story/" not in readme


def test_complex_constraint_sweep_published_representative_joint_plots():
    root = Path(__file__).resolve().parents[1]
    asset_dir = root / "assets" / "complex-constraint-sweep"
    expected = {
        "long_path_01_torque_speed_drop_joint_torque.svg": "Joint Torque",
        "long_path_01_torque_speed_drop_joint_jerk.svg": "Joint Jerk",
        "long_path_01_tight_jerk_joint_jerk.svg": "Joint Jerk",
        "long_path_01_tight_jerk_joint_torque_rate.svg": "Joint Torque Rate",
        "path_14_combined_torque_jerk_joint_torque.svg": "Joint Torque",
        "path_12_combined_torque_jerk_joint_acceleration.svg": "Joint Acceleration",
    }
    assert (asset_dir / "index.html").exists()
    for filename, title in expected.items():
        svg_path = asset_dir / filename
        assert svg_path.exists(), filename
        svg = svg_path.read_text(encoding="utf-8")
        assert title in svg
        assert "<polyline" in svg
