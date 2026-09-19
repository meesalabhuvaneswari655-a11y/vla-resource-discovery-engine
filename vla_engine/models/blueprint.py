"""Blueprint model for synthesized VLA training recipes."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from vla_engine.models.resource import Category, Resource
from vla_engine.models.query import ParsedQuery


class PipelineStep(BaseModel):
    step_number: int
    title: str
    description: str
    recommended_tools: List[str]
    inputs: str
    outputs: str
    hardware_notes: str


class VLABlueprint(BaseModel):
    query: ParsedQuery
    top_stack: Dict[Category, Resource] = Field(default_factory=dict)
    compatibility_score: float = 0.0
    compatibility_verdict: str = ""
    pipeline_steps: List[PipelineStep] = Field(default_factory=list)
    hardware_summary: Dict[str, str] = Field(default_factory=dict)
    all_ranked_resources: Dict[Category, List[Resource]] = Field(default_factory=dict)
    generated_at: str = ""
