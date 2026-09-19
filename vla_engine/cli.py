import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from vla_engine.discovery.aggregator import DiscoveryAggregator
from vla_engine.evaluation.ranker import ResourceRanker
from vla_engine.models.query import ScoringWeights
from vla_engine.parser.query_parser import QueryParser
from vla_engine.synthesis.blueprint_generator import BlueprintGenerator
from vla_engine.synthesis.formatter import ReportFormatter

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
})

console = Console(theme=custom_theme)


def run_search_and_rank(
    query_text: str,
    weights: ScoringWeights = None,
    use_live_apis: bool = True,
    export_format: str = None,
    output_file: str = None,
) -> None:
    """Execute end-to-end query parsing, discovery, evaluation, ranking, and synthesis."""
    console.print(
        Panel.fit(
            "[bold cyan]Intelligent VLA & Robotics AI Resource Discovery Engine[/bold cyan]\n"
            "[dim]Autonomous Search, Ingestion, Multi-Factor Scoring, and Blueprint Generation[/dim]",
            border_style="cyan",
        )
    )

    # 1. Parse Query
    with console.status("[bold green]Parsing natural language query and extracting intent...", spinner="dots"):
        parser = QueryParser(default_weights=weights)
        parsed_query = parser.parse(query_text, custom_weights=weights)

    # Print Intent Decomposition
    intent_table = Table(title="[bold]1. Query Intent Decomposition[/bold]", show_lines=True)
    intent_table.add_column("Dimension", style="cyan", width=24)
    intent_table.add_column("Parsed Values", style="white")

    intent_table.add_row("Raw Query", f"[italic]{parsed_query.raw_query}[/italic]")
    intent_table.add_row("Target Tasks", ", ".join(parsed_query.target_tasks) or "Object Manipulation")
    intent_table.add_row("Modalities", ", ".join(parsed_query.modalities))
    intent_table.add_row("Target Embodiments", ", ".join(parsed_query.target_embodiments) or "Multi-Embodiment (Franka, WidowX, ALOHA)")
    intent_table.add_row("Pillars Requested", f"{len(parsed_query.requested_categories)} Pillars")
    console.print(intent_table)
    console.print("")

    # 2. Multi-Source Discovery
    with console.status("[bold green]Querying multi-source providers (Hugging Face, GitHub, ArXiv, Curated KB)...", spinner="earth"):
        aggregator = DiscoveryAggregator(use_live_apis=use_live_apis)
        discovered_resources = aggregator.discover_all(parsed_query)

    console.print(f"[success][OK] Discovered and normalized [bold]{len(discovered_resources)}[/bold] relevant candidates across providers.[/success]\n")

    # 3. Multi-Factor Evaluation & Ranking
    with console.status("[bold green]Executing multi-dimensional scoring and stack compatibility evaluation...", spinner="bouncingBar"):
        ranker = ResourceRanker()
        ranked_categories = ranker.rank_resources(discovered_resources, parsed_query, top_k_per_category=4)

        generator = BlueprintGenerator()
        blueprint = generator.generate(ranked_categories, parsed_query)

    # 4. Display Recommended Stack
    stack_table = Table(
        title=f"[bold]2. Optimal VLA Technical Stack (Compatibility Score: {blueprint.compatibility_score}/100)[/bold]",
        show_lines=True,
    )
    stack_table.add_column("Pillar", style="cyan", width=22)
    stack_table.add_column("Top Ranked Asset", style="bold green", width=24)
    stack_table.add_column("Score", justify="center", style="bold yellow", width=10)
    stack_table.add_column("License", justify="center", style="dim", width=14)
    stack_table.add_column("Key Advantage / Differentiator", style="white")

    for cat, res in blueprint.top_stack.items():
        stack_table.add_row(
            cat.display_name,
            res.name,
            f"{res.final_score:.1f}",
            res.license,
            res.summary[:85] + ("..." if len(res.summary) > 85 else ""),
        )

    console.print(stack_table)
    console.print(f"[dim]Verdict: {blueprint.compatibility_verdict}[/dim]\n")

    # 5. Display Category Rankings
    for cat, res_list in ranked_categories.items():
        cat_table = Table(title=f"[bold cyan]{cat.display_name} Leaderboard[/bold cyan]", show_lines=False)
        cat_table.add_column("Rank", justify="center", width=6)
        cat_table.add_column("Resource Name", style="bold white", width=26)
        cat_table.add_column("Composite", justify="center", style="bold yellow", width=10)
        cat_table.add_column("Relevance", justify="center", width=10)
        cat_table.add_column("Adoption", justify="center", width=10)
        cat_table.add_column("Repro", justify="center", width=10)
        cat_table.add_column("Freshness", justify="center", width=10)
        cat_table.add_column("Stars / DLs", justify="right", style="dim", width=18)

        for rank_idx, r in enumerate(res_list, 1):
            sc = r.scores
            if sc:
                cat_table.add_row(
                    f"#{rank_idx}",
                    r.name,
                    f"{sc.composite_score:.1f}",
                    f"{sc.semantic_relevance:.1f}",
                    f"{sc.community_adoption:.1f}",
                    f"{sc.reproducibility:.1f}",
                    f"{sc.freshness:.1f}",
                    f"*{r.github_stars:,} / DL:{r.hf_downloads:,}",
                )

        console.print(cat_table)
        console.print("")

    # 6. Display 4-Step Pipeline Summary
    pipeline_panel_content = []
    for step in blueprint.pipeline_steps:
        pipeline_panel_content.append(
            f"[bold green]Step {step.step_number}: {step.title}[/bold green]\n"
            f"[white]{step.description}[/white]\n"
            f"[dim]Tools: {', '.join(step.recommended_tools)} | HW: {step.hardware_notes}[/dim]\n"
        )

    console.print(
        Panel(
            "\n".join(pipeline_panel_content),
            title="[bold yellow]3. Synthesized End-to-End VLA Training Roadmap[/bold yellow]",
            border_style="yellow",
        )
    )

    # 7. Export if requested
    if export_format:
        fmt = export_format.lower()
        if fmt == "markdown" or fmt == "md":
            output_content = ReportFormatter.to_markdown(blueprint)
            default_name = "vla_evaluation_report.md"
        elif fmt == "json":
            output_content = ReportFormatter.to_json(blueprint)
            default_name = "vla_evaluation_report.json"
        else:
            console.print(f"[warning]Unsupported export format: {export_format}[/warning]")
            return

        target_path = Path(output_file if output_file else default_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(output_content)

        console.print(f"[success][OK] Successfully exported full evaluation blueprint to: [bold]{target_path.resolve()}[/bold][/success]")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Search & Ranking Engine for VLM & Robotics AI Resources."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=(
            "Find the best open-source datasets, VLM architectures, robotics datasets, "
            "simulation environments, and training frameworks for developing a vision-language-action "
            "model for a robot capable of object manipulation and natural-language instruction following."
        ),
        help="Natural language search query",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live API calls and use curated seed database only",
    )
    parser.add_argument(
        "--export",
        "-e",
        choices=["markdown", "json", "md"],
        default=None,
        help="Export report format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for export",
    )
    parser.add_argument(
        "--weight-relevance",
        type=float,
        default=0.35,
        help="Weight for semantic relevance [0.0 - 1.0]",
    )
    parser.add_argument(
        "--weight-adoption",
        type=float,
        default=0.20,
        help="Weight for stars/downloads/citations [0.0 - 1.0]",
    )
    parser.add_argument(
        "--weight-reproducibility",
        type=float,
        default=0.20,
        help="Weight for checkpoints & permissive license [0.0 - 1.0]",
    )
    parser.add_argument(
        "--weight-freshness",
        type=float,
        default=0.15,
        help="Weight for recency [0.0 - 1.0]",
    )
    parser.add_argument(
        "--weight-compatibility",
        type=float,
        default=0.10,
        help="Weight for ecosystem stack compatibility [0.0 - 1.0]",
    )

    args = parser.parse_args()

    weights = ScoringWeights(
        semantic_relevance=args.weight_relevance,
        community_adoption=args.weight_adoption,
        reproducibility=args.weight_reproducibility,
        freshness=args.weight_freshness,
        compatibility=args.weight_compatibility,
    )

    run_search_and_rank(
        query_text=args.query,
        weights=weights,
        use_live_apis=not args.no_live,
        export_format=args.export,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
