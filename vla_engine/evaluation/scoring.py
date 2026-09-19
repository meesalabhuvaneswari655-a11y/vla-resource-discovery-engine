"""Mathematical multi-dimensional scoring engine for VLA & VLM resources."""

from datetime import datetime
import math
from typing import Dict, List
from vla_engine.models.query import ParsedQuery, ScoringWeights
from vla_engine.models.resource import LicenseType, Resource, ScoreBreakdown


class ScoringEngine:
    """Computes normalized multi-factor scores for candidate resources."""

    def __init__(self, current_year: int = 2024, current_month: int = 10):
        self.current_year = current_year
        self.current_month = current_month

    def score_resource(
        self,
        resource: Resource,
        query: ParsedQuery,
        compatibility_score: float = 85.0,
    ) -> ScoreBreakdown:
        """Calculate complete score breakdown for a resource."""
        weights = query.weights.normalize()

        s_rel = self._compute_semantic_relevance(resource, query)
        s_adopt = self._compute_adoption(resource)
        s_repro = self._compute_reproducibility(resource)
        s_fresh = self._compute_freshness(resource)
        s_compat = compatibility_score

        composite = (
            weights.semantic_relevance * s_rel
            + weights.community_adoption * s_adopt
            + weights.reproducibility * s_repro
            + weights.freshness * s_fresh
            + weights.compatibility * s_compat
        )

        # Clamp composite score to [0.0, 100.0]
        composite = max(0.0, min(100.0, round(composite, 2)))

        return ScoreBreakdown(
            semantic_relevance=round(s_rel, 2),
            community_adoption=round(s_adopt, 2),
            reproducibility=round(s_repro, 2),
            freshness=round(s_fresh, 2),
            compatibility=round(s_compat, 2),
            composite_score=composite,
        )

    def _compute_semantic_relevance(self, resource: Resource, query: ParsedQuery) -> float:
        """Compute semantic and domain relevance score [0 - 100]."""
        searchable_text = " ".join([
            resource.name,
            resource.summary,
            resource.description,
            " ".join(resource.target_tasks),
            " ".join(resource.modalities),
            " ".join(resource.tags),
            " ".join(resource.supported_embodiments),
            resource.action_space or "",
        ]).lower()

        score = 25.0  # Base prior

        # 1. Task Match (+35 max)
        for task in query.target_tasks:
            task_lower = task.lower()
            if task_lower in searchable_text:
                score += 20.0
            else:
                # Check token overlap
                task_tokens = task_lower.split()
                if any(tok in searchable_text for tok in task_tokens):
                    score += 10.0

        # 2. Modality Match (+20 max)
        for mod in query.modalities:
            if mod.lower() in searchable_text:
                score += 7.0

        # 3. Embodiment Match (+10 max)
        if query.target_embodiments:
            for emb in query.target_embodiments:
                if emb.lower() in searchable_text:
                    score += 10.0
                    break
        else:
            # If query didn't specify embodiment, grant partial credit for versatile generalist embodiments
            if any(e in searchable_text for e in ["franka", "widowx", "aloha", "ur5"]):
                score += 8.0

        # 4. Keyword tokens overlap (+10 max)
        matching_keywords = sum(1 for kw in query.keywords if kw in searchable_text)
        keyword_ratio = matching_keywords / max(1, len(query.keywords))
        score += keyword_ratio * 10.0

        return min(100.0, max(10.0, score))

    def _compute_adoption(self, resource: Resource) -> float:
        """Sublinear logarithmic scaling of traction metrics [0 - 100]."""
        # Benchmark ceilings
        max_stars = 25000.0
        max_downloads = 500000.0
        max_citations = 1000.0

        score_stars = math.log(1 + min(resource.github_stars, max_stars)) / math.log(1 + max_stars)
        score_dl = math.log(1 + min(resource.hf_downloads, max_downloads)) / math.log(1 + max_downloads)
        score_cite = math.log(1 + min(resource.citations, max_citations)) / math.log(1 + max_citations)

        # Weighted combination of metrics
        combined = (0.45 * score_stars) + (0.35 * score_dl) + (0.20 * score_cite)
        return min(100.0, max(10.0, combined * 100.0))

    def _compute_reproducibility(self, resource: Resource) -> float:
        """Score based on license permissiveness, checkpoints, and docs [0 - 100]."""
        score = 0.0

        # 1. License Permissiveness (up to 35 pts)
        if resource.license_type == LicenseType.PERMISSIVE:
            score += 35.0
        elif resource.license_type == LicenseType.NON_COMMERCIAL:
            score += 20.0
        elif resource.license_type == LicenseType.RESTRICTED:
            score += 10.0
        else:
            score += 15.0

        # 2. Checkpoints / Downloadable Weights (up to 30 pts)
        if resource.has_checkpoints or bool(resource.checkpoint_urls):
            score += 30.0

        # 3. Benchmark or Simulation Integration (up to 20 pts)
        if bool(resource.compatible_simulators) or bool(resource.compatible_frameworks):
            score += 20.0

        # 4. Hardware & Requirements Specification (up to 15 pts)
        if bool(resource.hardware_requirements):
            score += 15.0
        elif bool(resource.github_url) or bool(resource.paper_url):
            score += 10.0

        return min(100.0, max(5.0, score))

    def _compute_freshness(self, resource: Resource) -> float:
        """Exponential recency decay curve [0 - 100]."""
        if not resource.published_date:
            return 60.0

        try:
            parts = resource.published_date.split("-")
            pub_year = int(parts[0])
            pub_month = int(parts[1]) if len(parts) > 1 else 1
        except Exception:
            return 60.0

        # Age in months
        age_months = (self.current_year - pub_year) * 12 + (self.current_month - pub_month)
        age_months = max(0, age_months)

        # Exponential decay with half-life of 24 months
        # Score = 100 * exp(-lambda * age_months)
        decay_constant = math.log(2) / 24.0
        score = 100.0 * math.exp(-decay_constant * age_months)

        return min(100.0, max(20.0, score))
