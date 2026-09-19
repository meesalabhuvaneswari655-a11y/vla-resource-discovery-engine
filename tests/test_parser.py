"""Unit tests for query parsing and intent decomposition."""

import pytest
from vla_engine.models.resource import Category
from vla_engine.models.query import ScoringWeights
from vla_engine.parser.query_parser import QueryParser


def test_query_parser_prompt_query():
    parser = QueryParser()
    query = (
        "Find the best open-source datasets, VLM architectures, robotics datasets, "
        "simulation environments, and training frameworks for developing a vision-language-action "
        "model for a robot capable of object manipulation and natural-language instruction following."
    )
    parsed = parser.parse(query)

    assert "object manipulation" in parsed.target_tasks
    assert "instruction following" in parsed.target_tasks
    assert "vision" in parsed.modalities
    assert "language" in parsed.modalities
    assert "action" in parsed.modalities

    # Should detect all 5 categories
    assert Category.VLM_BACKBONE in parsed.requested_categories
    assert Category.VLA_POLICY in parsed.requested_categories
    assert Category.ROBOTICS_DATASET in parsed.requested_categories
    assert Category.SIMULATION in parsed.requested_categories
    assert Category.TRAINING_FRAMEWORK in parsed.requested_categories


def test_weights_normalization():
    weights = ScoringWeights(
        semantic_relevance=2.0,
        community_adoption=2.0,
        reproducibility=2.0,
        freshness=2.0,
        compatibility=2.0,
    ).normalize()

    assert pytest.approx(weights.semantic_relevance, 0.01) == 0.2
    assert pytest.approx(weights.community_adoption, 0.01) == 0.2
    assert pytest.approx(weights.reproducibility, 0.01) == 0.2
    assert pytest.approx(weights.freshness, 0.01) == 0.2
    assert pytest.approx(weights.compatibility, 0.01) == 0.2


def test_embodiment_detection():
    parser = QueryParser()
    parsed = parser.parse("Find best manipulation policy for ALOHA bimanual teleoperation setup")
    assert any("ALOHA" in e for e in parsed.target_embodiments)
