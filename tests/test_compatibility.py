"""Unit tests for ecosystem compatibility engine."""

from vla_engine.evaluation.compatibility import CompatibilityEngine
from vla_engine.models.resource import Category, Resource


def test_known_synergy_stack():
    engine = CompatibilityEngine()

    r_vlm = Resource(
        id="prismatic-vlms",
        name="Prismatic VLMs",
        category=Category.VLM_BACKBONE,
        summary="",
        description="",
    )
    r_vla = Resource(
        id="openvla-7b",
        name="OpenVLA",
        category=Category.VLA_POLICY,
        summary="",
        description="",
    )
    r_data = Resource(
        id="open-x-embodiment",
        name="Open X-Embodiment",
        category=Category.ROBOTICS_DATASET,
        summary="",
        description="",
    )
    r_sim = Resource(
        id="simpler-benchmark",
        name="SIMPLER",
        category=Category.SIMULATION,
        summary="",
        description="",
    )
    r_fw = Resource(
        id="openvla-codebase",
        name="OpenVLA Codebase",
        category=Category.TRAINING_FRAMEWORK,
        summary="",
        description="",
    )

    stack = {
        Category.VLM_BACKBONE: r_vlm,
        Category.VLA_POLICY: r_vla,
        Category.ROBOTICS_DATASET: r_data,
        Category.SIMULATION: r_sim,
        Category.TRAINING_FRAMEWORK: r_fw,
    }

    result = engine.evaluate_stack(stack)
    assert result["score"] >= 90.0
    assert "Optimal Synergistic Stack" in result["verdict"]
