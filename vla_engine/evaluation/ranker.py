"""Ranker orchestrator: scores, ranks, and categorizes candidate resources."""

from collections import defaultdict
from typing import Dict, List, Optional
from vla_engine.evaluation.compatibility import CompatibilityEngine
from vla_engine.evaluation.scoring import ScoringEngine
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, Resource


class ResourceRanker:
    """Evaluates and ranks candidate resources according to query intent and multi-factor criteria."""

    def __init__(
        self,
        scoring_engine: Optional[ScoringEngine] = None,
        compatibility_engine: Optional[CompatibilityEngine] = None,
    ):
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.compatibility_engine = compatibility_engine or CompatibilityEngine()

    def rank_resources(
        self,
        resources: List[Resource],
        query: ParsedQuery,
        top_k_per_category: int = 5,
    ) -> Dict[Category, List[Resource]]:
        """Score and rank resources, grouped by category."""
        # 1. First pass: compute cross-resource compatibility
        scored_resources: List[Resource] = []
        for res in resources:
            compat_score = self.compatibility_engine.compute_compatibility(res, resources)
            score_breakdown = self.scoring_engine.score_resource(
                resource=res,
                query=query,
                compatibility_score=compat_score,
            )
            # Create a clone/copy with updated scores
            res_scored = res.model_copy()
            res_scored.scores = score_breakdown
            scored_resources.append(res_scored)

        # 2. Group by Category
        by_category: Dict[Category, List[Resource]] = defaultdict(list)
        for res in scored_resources:
            by_category[res.category].append(res)

        # 3. Sort each category by composite score descending
        ranked_categories: Dict[Category, List[Resource]] = {}
        target_categories = query.requested_categories if query.requested_categories else list(Category)

        for cat in target_categories:
            cat_list = by_category.get(cat, [])
            # Sort by composite score descending, tiebreak on stars + downloads
            sorted_cat = sorted(
                cat_list,
                key=lambda r: (
                    r.scores.composite_score if r.scores else 0.0,
                    r.github_stars + r.hf_downloads,
                ),
                reverse=True,
            )
            ranked_categories[cat] = sorted_cat[:top_k_per_category]

        return ranked_categories

    def get_overall_leaderboard(
        self,
        ranked_by_category: Dict[Category, List[Resource]],
        limit: int = 10,
    ) -> List[Resource]:
        """Flatten ranked categories into an overall top-K leaderboard."""
        all_res: List[Resource] = []
        for cat_list in ranked_by_category.values():
            all_res.extend(cat_list)

        return sorted(
            all_res,
            key=lambda r: r.scores.composite_score if r.scores else 0.0,
            reverse=True,
        )[:limit]
