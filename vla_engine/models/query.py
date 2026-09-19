"""Query representation and weighting profiles."""

from typing import List, Optional
from pydantic import BaseModel, Field
from vla_engine.models.resource import Category


class ScoringWeights(BaseModel):
    semantic_relevance: float = Field(0.35, description="Weight for semantic & ontology relevance [0.0 - 1.0]")
    community_adoption: float = Field(0.20, description="Weight for stars/downloads/citations [0.0 - 1.0]")
    reproducibility: float = Field(0.20, description="Weight for checkpoints & permissive license [0.0 - 1.0]")
    freshness: float = Field(0.15, description="Weight for recent publication/activity [0.0 - 1.0]")
    compatibility: float = Field(0.10, description="Weight for ecosystem stack compatibility [0.0 - 1.0]")

    def normalize(self) -> "ScoringWeights":
        total = (
            self.semantic_relevance
            + self.community_adoption
            + self.reproducibility
            + self.freshness
            + self.compatibility
        )
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            semantic_relevance=self.semantic_relevance / total,
            community_adoption=self.community_adoption / total,
            reproducibility=self.reproducibility / total,
            freshness=self.freshness / total,
            compatibility=self.compatibility / total,
        )


class ParsedQuery(BaseModel):
    raw_query: str
    target_tasks: List[str] = Field(default_factory=list)
    target_embodiments: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    requested_categories: List[Category] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    prefer_permissive_license: bool = True
    require_checkpoints: bool = False
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
