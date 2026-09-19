"""Live discovery provider for GitHub repositories."""

import os
import requests
from typing import List
from vla_engine.discovery.base import BaseDiscoveryProvider
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, LicenseType, Resource


class GitHubProvider(BaseDiscoveryProvider):
    """Discovers robotics and VLA repositories using GitHub Search API."""

    def __init__(self, api_token: str = None):
        super().__init__(name="github")
        self.api_token = api_token or os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VLA-Resource-Discovery-Engine/1.0",
        }
        if self.api_token:
            self.headers["Authorization"] = f"token {self.api_token}"

    def discover(self, query: ParsedQuery) -> List[Resource]:
        """Search GitHub for repositories matching query keywords."""
        search_queries = [
            "vision-language-action in:name,description,topics",
            "robotics manipulation dataset in:description,topics",
            "robotics simulation benchmark in:description,topics",
        ]

        results: List[Resource] = []
        for sq in search_queries[:2]:
            repos = self._search_github(sq)
            results.extend(repos)

        return results

    def _search_github(self, search_query: str) -> List[Resource]:
        cache_id = f"gh_{search_query}"
        cached = self._read_cache(cache_id)
        if cached is not None:
            return [Resource(**item) for item in cached]

        url = "https://api.github.com/search/repositories"
        params = {
            "q": search_query,
            "sort": "stars",
            "order": "desc",
            "per_page": 5,
        }

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=6)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        items = data.get("items", [])
        resources: List[Resource] = []

        for item in items:
            name = item.get("name", "")
            full_name = item.get("full_name", "")
            description = item.get("description") or "Open source robotics codebase"
            stars = item.get("stargazers_count", 0)
            license_info = item.get("license") or {}
            license_key = license_info.get("spdx_id", "Unknown") if isinstance(license_info, dict) else "Unknown"
            topics = item.get("topics", [])
            pushed_at = item.get("pushed_at")

            # Determine category heuristic
            name_lower = (full_name + " " + description).lower()
            if any(k in name_lower for k in ["sim", "isaac", "mujoco", "environment"]):
                cat = Category.SIMULATION
            elif any(k in name_lower for k in ["dataset", "demonstration", "trajectories"]):
                cat = Category.ROBOTICS_DATASET
            elif any(k in name_lower for k in ["framework", "toolkit", "train", "lerobot"]):
                cat = Category.TRAINING_FRAMEWORK
            elif any(k in name_lower for k in ["vla", "policy", "action"]):
                cat = Category.VLA_POLICY
            else:
                cat = Category.VLM_BACKBONE

            lic_type = LicenseType.UNKNOWN
            if any(l in license_key.lower() for l in ["mit", "apache", "bsd"]):
                lic_type = LicenseType.PERMISSIVE

            res = Resource(
                id=f"gh-{full_name.replace('/', '-')}",
                name=name,
                category=cat,
                summary=description[:150],
                description=description,
                organization=full_name.split("/")[0],
                url=item.get("html_url"),
                github_url=item.get("html_url"),
                license=license_key,
                license_type=lic_type,
                published_date=pushed_at[:10] if pushed_at else None,
                github_stars=stars,
                target_tasks=["robotics manipulation"],
                modalities=["Vision", "Action"],
                tags=topics[:8],
                source="github",
            )
            resources.append(res)

        self._write_cache(cache_id, [r.model_dump() for r in resources])
        return resources
