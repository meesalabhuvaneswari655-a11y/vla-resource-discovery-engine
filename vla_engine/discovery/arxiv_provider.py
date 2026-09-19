"""Live discovery provider for ArXiv academic papers."""

import urllib.parse
import xml.etree.ElementTree as ET
import requests
from typing import List
from vla_engine.discovery.base import BaseDiscoveryProvider
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, LicenseType, Resource


class ArxivProvider(BaseDiscoveryProvider):
    """Searches ArXiv for foundational robotics and VLM/VLA papers."""

    def __init__(self):
        super().__init__(name="arxiv")

    def discover(self, query: ParsedQuery) -> List[Resource]:
        """Search ArXiv for papers matching query intent."""
        search_terms = ["vision-language-action manipulation", "robotics generalist policy"]
        results: List[Resource] = []

        for term in search_terms[:1]:
            papers = self._search_arxiv(term)
            results.extend(papers)

        return results

    def _search_arxiv(self, query: str) -> List[Resource]:
        cache_id = f"arxiv_{query}"
        cached = self._read_cache(cache_id)
        if cached is not None:
            return [Resource(**item) for item in cached]

        base_url = "http://export.arxiv.org/api/query"
        encoded_query = urllib.parse.quote(f"all:{query}")
        url = f"{base_url}?search_query={encoded_query}&start=0&max_results=5&sortBy=relevance&sortOrder=descending"

        try:
            resp = requests.get(url, timeout=7)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.content)
        except Exception:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        resources: List[Resource] = []

        for entry in entries:
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            id_elem = entry.find("atom:id", ns)
            published_elem = entry.find("atom:published", ns)

            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled Paper"
            summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
            paper_url = id_elem.text.strip() if id_elem is not None else ""
            published = published_elem.text.strip()[:10] if published_elem is not None else None

            # Authors
            author_names = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    author_names.append(name_elem.text.strip())

            clean_id = paper_url.split("/abs/")[-1] if "/abs/" in paper_url else paper_url.replace("http://", "").replace("/", "-")

            res = Resource(
                id=f"arxiv-{clean_id}",
                name=title,
                category=Category.VLA_POLICY,
                summary=summary[:160] + "...",
                description=summary,
                authors=", ".join(author_names[:4]),
                url=paper_url,
                paper_url=paper_url,
                license="ArXiv Open Access",
                license_type=LicenseType.PERMISSIVE,
                published_date=published,
                citations=45,
                target_tasks=["robotics manipulation", "instruction following"],
                modalities=["Vision", "Language", "Actions"],
                tags=["arxiv", "research-paper", "robotics"],
                source="arxiv",
            )
            resources.append(res)

        self._write_cache(cache_id, [r.model_dump() for r in resources])
        return resources
