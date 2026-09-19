"""Unit tests for blueprint generation and serialization."""

from vla_engine.discovery.aggregator import DiscoveryAggregator
from vla_engine.evaluation.ranker import ResourceRanker
from vla_engine.parser.query_parser import QueryParser
from vla_engine.synthesis.blueprint_generator import BlueprintGenerator
from vla_engine.synthesis.formatter import ReportFormatter


def test_end_to_end_blueprint_flow():
    parser = QueryParser()
    query = (
        "Find the best open-source datasets, VLM architectures, robotics datasets, "
        "simulation environments, and training frameworks for developing a vision-language-action "
        "model for a robot capable of object manipulation and natural-language instruction following."
    )
    parsed = parser.parse(query)

    aggregator = DiscoveryAggregator(use_live_apis=False)
    resources = aggregator.discover_all(parsed)

    ranker = ResourceRanker()
    ranked_cats = ranker.rank_resources(resources, parsed, top_k_per_category=3)

    generator = BlueprintGenerator()
    blueprint = generator.generate(ranked_cats, parsed)

    # Validate blueprint properties
    assert len(blueprint.top_stack) >= 4
    assert len(blueprint.pipeline_steps) == 4
    assert blueprint.compatibility_score > 70.0

    # Validate Markdown and JSON outputs
    md_report = ReportFormatter.to_markdown(blueprint)
    assert "# Vision-Language-Action (VLA) Model Development Blueprint" in md_report
    assert "OpenVLA" in md_report

    json_report = ReportFormatter.to_json(blueprint)
    assert "pipeline_steps" in json_report
