"""Provider for verified, curated seed knowledge base of VLA & VLM resources."""

import json
from pathlib import Path
from typing import List, Optional
from vla_engine.discovery.base import BaseDiscoveryProvider
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, LicenseType, Resource


class CuratedProvider(BaseDiscoveryProvider):
    """Loads curated and verified VLA/VLM resources from disk."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="curated_kb")
        if data_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.data_path = base_dir / "data" / "seed_knowledge_base.json"
        else:
            self.data_path = Path(data_path)

    def discover(self, query: ParsedQuery) -> List[Resource]:
        """Load curated resources filtered by requested categories."""
        if not self.data_path.exists():
            return []

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception:
            return []

        raw_items = content.get("resources", [])
        results: List[Resource] = []

        allowed_categories = set(query.requested_categories) if query.requested_categories else set(Category)

        for item in raw_items:
            try:
                cat_str = item.get("category")
                try:
                    category = Category(cat_str)
                except ValueError:
                    continue

                if category not in allowed_categories:
                    continue

                lic_type = item.get("license_type", "unknown").lower()
                try:
                    license_type = LicenseType(lic_type)
                except ValueError:
                    license_type = LicenseType.UNKNOWN

                resource = Resource(
                    id=item.get("id"),
                    name=item.get("name"),
                    category=category,
                    summary=item.get("summary", ""),
                    description=item.get("description", ""),
                    authors=item.get("authors"),
                    organization=item.get("organization"),
                    url=item.get("url"),
                    github_url=item.get("github_url"),
                    hf_url=item.get("hf_url"),
                    paper_url=item.get("paper_url"),
                    license=item.get("license", "Unknown"),
                    license_type=license_type,
                    published_date=item.get("published_date"),
                    github_stars=int(item.get("github_stars", 0)),
                    hf_downloads=int(item.get("hf_downloads", 0)),
                    citations=int(item.get("citations", 0)),
                    has_checkpoints=bool(item.get("has_checkpoints", False)),
                    checkpoint_urls=item.get("checkpoint_urls", []),
                    supported_embodiments=item.get("supported_embodiments", []),
                    action_space=item.get("action_space"),
                    target_tasks=item.get("target_tasks", []),
                    modalities=item.get("modalities", []),
                    compatible_frameworks=item.get("compatible_frameworks", []),
                    compatible_simulators=item.get("compatible_simulators", []),
                    hardware_requirements=item.get("hardware_requirements", {}),
                    tags=item.get("tags", []),
                    source="curated",
                )
                results.append(resource)
            except Exception:
                continue

        return results
