"""Discovery module exports."""

from vla_engine.discovery.aggregator import DiscoveryAggregator
from vla_engine.discovery.arxiv_provider import ArxivProvider
from vla_engine.discovery.base import BaseDiscoveryProvider
from vla_engine.discovery.curated_provider import CuratedProvider
from vla_engine.discovery.github_provider import GitHubProvider
from vla_engine.discovery.hf_provider import HuggingFaceProvider

__all__ = [
    "BaseDiscoveryProvider",
    "CuratedProvider",
    "HuggingFaceProvider",
    "GitHubProvider",
    "ArxivProvider",
    "DiscoveryAggregator",
]
