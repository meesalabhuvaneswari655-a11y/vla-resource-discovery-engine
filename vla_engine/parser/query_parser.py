"""Natural language query parser and intent decomposer."""

import re
from typing import List, Optional
from vla_engine.models.resource import Category
from vla_engine.models.query import ParsedQuery, ScoringWeights
from vla_engine.parser.taxonomy import (
    CATEGORY_TRIGGERS,
    EMBODIMENT_ONTOLOGY,
    MODALITY_ONTOLOGY,
    TASK_ONTOLOGY,
)


class QueryParser:
    """Parses natural language requests into structured VLA search queries."""

    def __init__(self, default_weights: Optional[ScoringWeights] = None):
        self.default_weights = (default_weights or ScoringWeights()).normalize()

    def parse(self, query_text: str, custom_weights: Optional[ScoringWeights] = None) -> ParsedQuery:
        """Parse raw query into a structured ParsedQuery object."""
        normalized_text = query_text.lower().strip()

        # 1. Identify Target Tasks
        detected_tasks = []
        for task_name, keywords in TASK_ONTOLOGY.items():
            for kw in keywords:
                if kw in normalized_text:
                    detected_tasks.append(task_name.replace("_", " "))
                    break

        if not detected_tasks:
            # Fallback default for general queries
            detected_tasks.append("object manipulation")

        # 2. Identify Modalities
        detected_modalities = []
        for modality, keywords in MODALITY_ONTOLOGY.items():
            for kw in keywords:
                if kw in normalized_text:
                    detected_modalities.append(modality)
                    break

        if not detected_modalities:
            detected_modalities = ["vision", "language", "action"]

        # 3. Identify Target Embodiments
        detected_embodiments = []
        for embodiment_name, aliases in EMBODIMENT_ONTOLOGY.items():
            for alias in aliases:
                if alias in normalized_text:
                    detected_embodiments.append(embodiment_name)
                    break

        # 4. Identify Requested Categories
        detected_categories = set()
        for category, triggers in CATEGORY_TRIGGERS.items():
            for trigger in triggers:
                if trigger in normalized_text:
                    detected_categories.add(category)
                    break

        # If user asked for general VLA or didn't restrict to 1-2 categories, include all 5 pillars
        if not detected_categories or "vision-language-action" in normalized_text or "vla" in normalized_text:
            if len(detected_categories) < 3:
                detected_categories = set(Category)

        # 5. Extract significant keywords
        stop_words = {
            "a", "an", "the", "for", "and", "or", "in", "of", "to", "with", "is",
            "are", "best", "find", "developing", "capable", "model", "models"
        }
        raw_tokens = re.findall(r"\b[a-z0-9\-\_]{3,}\b", normalized_text)
        keywords = [t for t in raw_tokens if t not in stop_words]

        weights = (custom_weights or self.default_weights).normalize()

        return ParsedQuery(
            raw_query=query_text,
            target_tasks=list(set(detected_tasks)),
            target_embodiments=detected_embodiments,
            modalities=detected_modalities,
            requested_categories=list(detected_categories),
            keywords=keywords,
            prefer_permissive_license=True,
            weights=weights,
        )
