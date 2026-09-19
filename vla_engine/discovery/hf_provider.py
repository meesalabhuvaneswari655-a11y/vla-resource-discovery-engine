"""Live discovery provider for Hugging Face Hub models and datasets."""

import os
import requests
from typing import Any, Dict, List
from vla_engine.discovery.base import BaseDiscoveryProvider
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, LicenseType, Resource


class HuggingFaceProvider(BaseDiscoveryProvider):
    """Discovers robotics models and datasets from Hugging Face Hub."""

    def __init__(self, api_token: str = None):
        super().__init__(name="huggingface")
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.headers = {"User-Agent": "VLA-Resource-Discovery-Engine/1.0"}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

    def discover(self, query: ParsedQuery) -> List[Resource]:
        """Query Hugging Face for models and datasets relevant to VLA / robotics."""
        results: List[Resource] = []

        search_terms = ["vla", "lerobot", "open-x-embodiment", "robotics"]
        if "instruction following" in query.target_tasks:
            search_terms.append("vision-language")

        # 1. Fetch Models
        if any(c in query.requested_categories for c in [Category.VLA_POLICY, Category.VLM_BACKBONE, Category.TRAINING_FRAMEWORK]):
            for term in search_terms[:2]:
                models = self._fetch_hf_models(term)
                results.extend(models)

        # 2. Fetch Datasets
        if Category.ROBOTICS_DATASET in query.requested_categories:
            for term in ["robotics", "open-x-embodiment"]:
                datasets = self._fetch_hf_datasets(term)
                results.extend(datasets)

        return results

    def _fetch_hf_models(self, search_term: str) -> List[Resource]:
        cache_id = f"models_{search_term}"
        cached = self._read_cache(cache_id)
        if cached is not None:
            return [Resource(**item) for item in cached]

        url = "https://huggingface.co/api/models"
        params = {
            "search": search_term,
            "sort": "downloads",
            "direction": -1,
            "limit": 8,
        }

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=6)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        resources: List[Resource] = []
        for item in data:
            model_id = item.get("id", "")
            tags = item.get("tags", [])
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)

            # Categorize based on tags
            if any(t in tags for t in ["robotics", "vla", "action"]):
                cat = Category.VLA_POLICY
            elif any(t in tags for t in ["vision-language", "image-text-to-text"]):
                cat = Category.VLM_BACKBONE
            else:
                cat = Category.VLA_POLICY

            # Determine license
            license_val = "unknown"
            for t in tags:
                if t.startswith("license:"):
                    license_val = t.replace("license:", "")
                    break

            res = Resource(
                id=f"hf-{model_id.replace('/', '-')}",
                name=model_id,
                category=cat,
                summary=f"Hugging Face model: {model_id} with {downloads:,} downloads and {likes} likes.",
                description=f"Model repository hosted on Hugging Face Hub under id '{model_id}'. Tagged with: {', '.join(tags[:6])}.",
                organization=model_id.split("/")[0] if "/" in model_id else "Community",
                url=f"https://huggingface.co/{model_id}",
                hf_url=f"https://huggingface.co/{model_id}",
                license=license_val,
                license_type=LicenseType.PERMISSIVE if any(l in license_val.lower() for l in ["mit", "apache", "bsd"]) else LicenseType.UNKNOWN,
                hf_downloads=downloads,
                has_checkpoints=True,
                checkpoint_urls=[f"https://huggingface.co/{model_id}"],
                target_tasks=["robotics manipulation", "vision-language"],
                modalities=["RGB Vision", "Language", "Robot Action"],
                tags=tags[:10],
                source="huggingface",
            )
            resources.append(res)

        # Cache results
        self._write_cache(cache_id, [r.model_dump() for r in resources])
        return resources

    def _fetch_hf_datasets(self, search_term: str) -> List[Resource]:
        cache_id = f"datasets_{search_term}"
        cached = self._read_cache(cache_id)
        if cached is not None:
            return [Resource(**item) for item in cached]

        url = "https://huggingface.co/api/datasets"
        params = {
            "search": search_term,
            "sort": "downloads",
            "direction": -1,
            "limit": 6,
        }

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=6)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        resources: List[Resource] = []
        for item in data:
            ds_id = item.get("id", "")
            tags = item.get("tags", [])
            downloads = item.get("downloads", 0)

            license_val = "unknown"
            for t in tags:
                if t.startswith("license:"):
                    license_val = t.replace("license:", "")
                    break

            res = Resource(
                id=f"hf-ds-{ds_id.replace('/', '-')}",
                name=f"{ds_id} (Dataset)",
                category=Category.ROBOTICS_DATASET,
                summary=f"Robotics dataset hosted on Hugging Face with {downloads:,} downloads.",
                description=f"Demonstrations / trajectory dataset '{ds_id}' on Hugging Face Hub.",
                organization=ds_id.split("/")[0] if "/" in ds_id else "Community",
                url=f"https://huggingface.co/datasets/{ds_id}",
                hf_url=f"https://huggingface.co/datasets/{ds_id}",
                license=license_val,
                license_type=LicenseType.PERMISSIVE if any(l in license_val.lower() for l in ["mit", "apache", "bsd", "cc-by"]) else LicenseType.UNKNOWN,
                hf_downloads=downloads,
                target_tasks=["manipulation demonstrations", "trajectory replay"],
                modalities=["RGB Vision", "Robot Proprioception", "Actions"],
                tags=tags[:10],
                source="huggingface",
            )
            resources.append(res)

        self._write_cache(cache_id, [r.model_dump() for r in resources])
        return resources
