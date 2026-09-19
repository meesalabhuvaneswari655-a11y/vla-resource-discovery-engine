"""Unit tests for discovery providers and aggregation."""

from vla_engine.discovery.aggregator import DiscoveryAggregator
from vla_engine.discovery.curated_provider import CuratedProvider
from vla_engine.models.resource import Category
from vla_engine.parser.query_parser import QueryParser


def test_curated_provider_loads_resources():
    provider = CuratedProvider()
    parser = QueryParser()
    query = parser.parse("robotics datasets and simulation")

    resources = provider.discover(query)
    assert len(resources) > 0

    # Ensure Open X-Embodiment and ManiSkill are present
    ids = [r.id for r in resources]
    assert "open-x-embodiment" in ids
    assert "maniskill3" in ids


def test_discovery_aggregator_deduplication():
    # Test offline aggregator
    aggregator = DiscoveryAggregator(use_live_apis=False)
    parser = QueryParser()
    query = parser.parse("vla policy for manipulation")

    resources = aggregator.discover_all(query)
    assert len(resources) > 0

    # Check no duplicate IDs
    ids = [r.id for r in resources]
    assert len(ids) == len(set(ids))
