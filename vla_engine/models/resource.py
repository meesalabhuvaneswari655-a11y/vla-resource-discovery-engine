"""Resource entity and category enumerations."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    VLM_BACKBONE = "vlm_backbone"
    VLA_POLICY = "vla_policy"
    ROBOTICS_DATASET = "robotics_dataset"
    SIMULATION = "simulation"
    TRAINING_FRAMEWORK = "training_framework"

    @property
    def display_name(self) -> str:
        mapping = {
            Category.VLM_BACKBONE: "VLM Foundation Architectures",
            Category.VLA_POLICY: "VLA Policies & Action Heads",
            Category.ROBOTICS_DATASET: "Robotics Manipulation Datasets",
            Category.SIMULATION: "Simulation Environments & Benchmarks",
            Category.TRAINING_FRAMEWORK: "Training Frameworks & Toolkits",
        }
        return mapping.get(self, self.value)


class LicenseType(str, Enum):
    PERMISSIVE = "permissive"         # MIT, Apache-2.0, BSD
    NON_COMMERCIAL = "non_commercial" # CC-BY-NC, Academic only
    RESTRICTED = "restricted"         # Proprietary / gated weights
    UNKNOWN = "unknown"


class ScoreBreakdown(BaseModel):
    semantic_relevance: float = Field(0.0, description="Relevance to query intent [0-100]")
    community_adoption: float = Field(0.0, description="Stars, downloads, citations [0-100]")
    reproducibility: float = Field(0.0, description="Checkpoints, docs, permissive license [0-100]")
    freshness: float = Field(0.0, description="Recency and maintenance velocity [0-100]")
    compatibility: float = Field(0.0, description="Interoperability with other stack components [0-100]")
    composite_score: float = Field(0.0, description="Weighted composite score [0-100]")


class Resource(BaseModel):
    id: str
    name: str
    category: Category
    summary: str
    description: str
    authors: Optional[str] = None
    organization: Optional[str] = None
    url: Optional[str] = None
    github_url: Optional[str] = None
    hf_url: Optional[str] = None
    paper_url: Optional[str] = None
    license: str = "Unknown"
    license_type: LicenseType = LicenseType.UNKNOWN
    published_date: Optional[str] = None
    github_stars: int = 0
    hf_downloads: int = 0
    citations: int = 0
    has_checkpoints: bool = False
    checkpoint_urls: List[str] = Field(default_factory=list)
    supported_embodiments: List[str] = Field(default_factory=list)
    action_space: Optional[str] = None
    target_tasks: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    compatible_frameworks: List[str] = Field(default_factory=list)
    compatible_simulators: List[str] = Field(default_factory=list)
    hardware_requirements: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    source: str = "curated"  # "curated", "huggingface", "github", "arxiv"
    scores: Optional[ScoreBreakdown] = None

    @property
    def final_score(self) -> float:
        return self.scores.composite_score if self.scores else 0.0
