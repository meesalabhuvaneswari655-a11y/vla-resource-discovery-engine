"""Quickstart script to run the VLA resource discovery and ranking pipeline."""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vla_engine.cli import run_search_and_rank
from vla_engine.models.query import ScoringWeights

if __name__ == "__main__":
    query = (
        "Find the best open-source datasets, VLM architectures, robotics datasets, "
        "simulation environments, and training frameworks for developing a vision-language-action "
        "model for a robot capable of object manipulation and natural-language instruction following."
    )

    output_path = root_dir / "examples" / "sample_evaluation_report.md"

    print("Running VLA Resource Discovery & Ranking Engine...")
    print(f"Target Query: {query}\n")

    run_search_and_rank(
        query_text=query,
        weights=ScoringWeights(
            semantic_relevance=0.35,
            community_adoption=0.20,
            reproducibility=0.20,
            freshness=0.15,
            compatibility=0.10,
        ),
        use_live_apis=True,
        export_format="markdown",
        output_file=str(output_path),
    )

    print(f"\nExecution complete! Report saved to: {output_path}")
