"""Unit tests for scoring algorithms."""

import pytest
from vla_engine.evaluation.scoring import ScoringEngine
from vla_engine.models.query import ParsedQuery, ScoringWeights
from vla_engine.models.resource import Category, LicenseType, Resource


@pytest.fixture
def sample_resource():
    return Resource(
        id="test-vla",
        name="Test VLA Policy",
        category=Category.VLA_POLICY,
        summary="A test vision-language-action policy for object manipulation and instruction following.",
        description="Detailed description for testing purposes.",
        license="Apache-2.0",
        license_type=LicenseType.PERMISSIVE,
        published_date="2024-06-01",
        github_stars=5000,
        hf_downloads=50000,
        citations=120,
        has_checkpoints=True,
        supported_embodiments=["Franka Panda"],
        target_tasks=["object manipulation", "instruction following"],
        modalities=["Vision", "Language", "Action"],
    )


@pytest.fixture
def sample_query():
    return ParsedQuery(
        raw_query="object manipulation instruction following",
        target_tasks=["object manipulation", "instruction following"],
        modalities=["vision", "language", "action"],
        target_embodiments=["Franka Panda"],
        requested_categories=[Category.VLA_POLICY],
        keywords=["manipulation", "instruction"],
        weights=ScoringWeights(),
    )


def test_scoring_engine_bounds(sample_resource, sample_query):
    engine = ScoringEngine(current_year=2024, current_month=10)
    score_breakdown = engine.score_resource(sample_resource, sample_query)

    assert 0.0 <= score_breakdown.semantic_relevance <= 100.0
    assert 0.0 <= score_breakdown.community_adoption <= 100.0
    assert 0.0 <= score_breakdown.reproducibility <= 100.0
    assert 0.0 <= score_breakdown.freshness <= 100.0
    assert 0.0 <= score_breakdown.compatibility <= 100.0
    assert 0.0 <= score_breakdown.composite_score <= 100.0

    # High quality resource with matching intent should have high composite score
    assert score_breakdown.composite_score > 75.0


def test_reproducibility_license_boost(sample_resource, sample_query):
    engine = ScoringEngine()
    
    # Permissive
    score_perm = engine._compute_reproducibility(sample_resource)
    
    # Restricted
    sample_resource.license_type = LicenseType.RESTRICTED
    score_restr = engine._compute_reproducibility(sample_resource)

    assert score_perm > score_restr
