"""Cross-pillar ecosystem compatibility matrix and evaluation engine."""

from typing import Dict, List, Optional, Set
from vla_engine.models.resource import Category, Resource


class CompatibilityEngine:
    """Evaluates whether components can be integrated into an end-to-end VLA stack."""

    # Explicit cross-resource compatibility pairings
    KNOWN_COMPATIBILITIES = {
        "openvla-7b": {
            "prismatic-vlms": 100.0,
            "open-x-embodiment": 100.0,
            "bridge-data-v2": 95.0,
            "droid-dataset": 90.0,
            "simpler-benchmark": 100.0,
            "maniskill3": 95.0,
            "isaac-lab": 90.0,
            "openvla-codebase": 100.0,
            "lerobot-huggingface": 95.0,
        },
        "openvla-codebase": {
            "openvla-7b": 100.0,
            "prismatic-vlms": 100.0,
            "open-x-embodiment": 100.0,
            "bridge-data-v2": 95.0,
            "simpler-benchmark": 100.0,
            "maniskill3": 95.0,
        },
        "prismatic-vlms": {
            "openvla-7b": 100.0,
            "openvla-codebase": 100.0,
            "open-x-embodiment": 95.0,
            "simpler-benchmark": 95.0,
            "lerobot-huggingface": 95.0,
        },
        "open-x-embodiment": {
            "openvla-7b": 100.0,
            "openvla-codebase": 100.0,
            "simpler-benchmark": 100.0,
            "maniskill3": 95.0,
            "lerobot-huggingface": 95.0,
            "octo-base": 100.0,
        },
        "simpler-benchmark": {
            "openvla-7b": 100.0,
            "openvla-codebase": 100.0,
            "open-x-embodiment": 100.0,
            "octo-base": 95.0,
            "lerobot-huggingface": 90.0,
        },
        "octo-base": {
            "open-x-embodiment": 100.0,
            "bridge-data-v2": 95.0,
            "simpler-benchmark": 95.0,
            "maniskill3": 90.0,
            "robocasa-simulator": 90.0,
            "lerobot-huggingface": 90.0,
        },
        "act-action-chunking": {
            "aloha-mobile-dataset": 100.0,
            "lerobot-huggingface": 100.0,
            "robomimic": 95.0,
            "maniskill3": 90.0,
            "isaac-lab": 88.0,
        },
        "diffusion-policy": {
            "robomimic": 100.0,
            "lerobot-huggingface": 100.0,
            "maniskill3": 95.0,
            "bridge-data-v2": 90.0,
        },
        "pi0-flow-matching": {
            "open-x-embodiment": 95.0,
            "droid-dataset": 95.0,
            "aloha-mobile-dataset": 90.0,
            "isaac-lab": 95.0,
            "maniskill3": 92.0,
            "lerobot-huggingface": 90.0,
        },
        "lerobot-huggingface": {
            "open-x-embodiment": 95.0,
            "bridge-data-v2": 95.0,
            "aloha-mobile-dataset": 100.0,
            "droid-dataset": 90.0,
            "maniskill3": 95.0,
            "isaac-lab": 90.0,
            "simpler-benchmark": 90.0,
            "paligemma": 95.0,
        },
        "maniskill3": {
            "open-x-embodiment": 95.0,
            "openvla-7b": 95.0,
            "act-action-chunking": 90.0,
            "lerobot-huggingface": 95.0,
            "robomimic": 90.0,
        },
    }

    def compute_compatibility(self, resource: Resource, candidate_pool: List[Resource]) -> float:
        """Compute compatibility score for a resource against other candidate resources."""
        if not candidate_pool:
            return 80.0

        res_id = resource.id
        scores = []

        # Check explicit pairings
        known_pairings = self.KNOWN_COMPATIBILITIES.get(res_id, {})

        for other in candidate_pool:
            if other.id == res_id or other.category == resource.category:
                continue

            other_id = other.id

            # Check direct or reciprocal pairing
            if other_id in known_pairings:
                scores.append(known_pairings[other_id])
            elif other_id in self.KNOWN_COMPATIBILITIES and res_id in self.KNOWN_COMPATIBILITIES[other_id]:
                scores.append(self.KNOWN_COMPATIBILITIES[other_id][res_id])
            else:
                # Check tag and framework overlap
                score = self._compute_heuristic_compatibility(resource, other)
                scores.append(score)

        if not scores:
            return 80.0

        return sum(scores) / len(scores)

    def _compute_heuristic_compatibility(self, r1: Resource, r2: Resource) -> float:
        """Heuristic compatibility based on common embodiments and frameworks."""
        base_score = 70.0

        # Shared embodiments
        emb1 = set(e.lower() for e in r1.supported_embodiments)
        emb2 = set(e.lower() for e in r2.supported_embodiments)
        if emb1 and emb2 and (emb1 & emb2):
            base_score += 15.0

        # Shared framework / simulator compatibility tags
        fw1 = set(f.lower() for f in r1.compatible_frameworks)
        fw2 = set(f.lower() for f in r2.compatible_frameworks)
        if fw1 and fw2 and (fw1 & fw2):
            base_score += 10.0

        sim1 = set(s.lower() for s in r1.compatible_simulators)
        sim2 = set(s.lower() for s in r2.compatible_simulators)
        if sim1 and sim2 and (sim1 & sim2):
            base_score += 5.0

        return min(95.0, base_score)

    def evaluate_stack(self, stack: Dict[Category, Resource]) -> Dict[str, any]:
        """Evaluate overall coherence of a selected 5-pillar VLA stack."""
        resources = list(stack.values())
        if len(resources) < 2:
            return {"score": 85.0, "verdict": "Single resource stack", "notes": []}

        pair_scores = []
        notes = []

        for i in range(len(resources)):
            for j in range(i + 1, len(resources)):
                r1 = resources[i]
                r2 = resources[j]
                
                score = 75.0
                if r1.id in self.KNOWN_COMPATIBILITIES and r2.id in self.KNOWN_COMPATIBILITIES[r1.id]:
                    score = self.KNOWN_COMPATIBILITIES[r1.id][r2.id]
                elif r2.id in self.KNOWN_COMPATIBILITIES and r1.id in self.KNOWN_COMPATIBILITIES[r2.id]:
                    score = self.KNOWN_COMPATIBILITIES[r2.id][r1.id]
                else:
                    score = self._compute_heuristic_compatibility(r1, r2)

                pair_scores.append(score)

        avg_score = sum(pair_scores) / len(pair_scores)

        if avg_score >= 92.0:
            verdict = "Optimal Synergistic Stack: Direct out-of-the-box integration verified."
        elif avg_score >= 82.0:
            verdict = "Highly Compatible Stack: Standard format conversions supported."
        else:
            verdict = "Moderately Compatible Stack: Custom adapters or data loaders required."

        return {
            "score": round(avg_score, 1),
            "verdict": verdict,
            "pair_count": len(pair_scores),
        }
