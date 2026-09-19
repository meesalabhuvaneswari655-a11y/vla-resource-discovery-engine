"""Aggregates and deduplicates resources from all discovery providers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Dict, List, Optional
from vla_engine.discovery.arxiv_provider import ArxivProvider
from vla_engine.discovery.curated_provider import CuratedProvider
from vla_engine.discovery.github_provider import GitHubProvider
from vla_engine.discovery.hf_provider import HuggingFaceProvider
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Resource


class DiscoveryAggregator:
    """Orchestrates multi-source resource discovery and entity deduplication."""

    def __init__(
        self,
        use_live_apis: bool = True,
        hf_token: Optional[str] = None,
        github_token: Optional[str] = None,
    ):
        self.providers = [CuratedProvider()]
        if use_live_apis:
            self.providers.extend([
                HuggingFaceProvider(api_token=hf_token),
                GitHubProvider(api_token=github_token),
                ArxivProvider(),
            ])

    def discover_all(self, query: ParsedQuery) -> List[Resource]:
        """Query all providers in parallel and deduplicate results."""
        all_candidates: List[Resource] = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_provider = {
                executor.submit(p.discover, query): p.name for p in self.providers
            }
            for future in as_completed(future_to_provider):
                p_name = future_to_provider[future]
                try:
                    res_list = future.result()
                    all_candidates.extend(res_list)
                except Exception:
                    pass

        return self._deduplicate(all_candidates)

    def _normalize_name(self, name: str) -> str:
        """Normalize resource name for deduplication."""
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
        return cleaned

    def _deduplicate(self, resources: List[Resource]) -> List[Resource]:
        """Merge duplicate resources, prioritizing curated entries."""
        unique_by_key: Dict[str, Resource] = {}

        # Sort so that 'curated' entries are processed first and preserved
        sorted_resources = sorted(
            resources,
            key=lambda r: (0 if r.source == "curated" else 1, -r.github_stars, -r.hf_downloads)
        )

        for res in sorted_resources:
            norm_name = self._normalize_name(res.name)
            
            # Check if an existing entry shares the normalized name
            match_found = False
            for key, existing in unique_by_key.items():
                if norm_name in key or key in norm_name or (res.github_url and res.github_url == existing.github_url):
                    # Merge metadata if existing has missing fields
                    if not existing.hf_downloads and res.hf_downloads:
                        existing.hf_downloads = res.hf_downloads
                    if not existing.github_stars and res.github_stars:
                        existing.github_stars = res.github_stars
                    match_found = True
                    break

            if not match_found:
                unique_by_key[norm_name] = res

        return list(unique_by_key.values())
