"""
Collaborative filtering based on similar researchers
"""

import logging
from typing import Dict, List, Set

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    def __init__(self, researcher_profile: Dict):
        self.profile = researcher_profile
        self.similar_researchers = self._find_similar_researchers()

    def _find_similar_researchers(self) -> List[Dict]:
        """Find researchers with similar interests"""
        similar = []

        # Based on co-authorship
        if self.profile.get("scholar_data"):
            coauthors = set()
            for paper in self.profile["scholar_data"].get("papers", []):
                if "authors" in paper:
                    coauthors.update(paper["authors"])

            # Remove self
            coauthors.discard(self.profile["scholar_data"].get("name"))

            similar.extend(
                [
                    {"name": author, "type": "coauthor"}
                    for author in list(coauthors)[:10]
                ]
            )

        # Based on research interests (hardcoded for now, could use API)
        interest_based = [
            {"name": "Sebastian Ruder", "interests": ["optimization", "NLP"]},
            {"name": "Kingma", "interests": ["optimization", "Adam"]},
            {
                "name": "Geoffrey Hinton",
                "interests": ["neural networks", "optimization"],
            },
        ]

        similar.extend(interest_based)

        return similar

    def get_collaborative_recommendations(self, papers: List[Dict]) -> List[Dict]:
        """Score papers based on what similar researchers are reading/citing"""

        # For now, boost papers that mention optimization methods
        # In production, this would check citation patterns of similar researchers

        for paper in papers:
            collaborative_score = 0.0

            # Check if paper is by a similar researcher
            paper_authors = set(paper.get("authors", []))
            for researcher in self.similar_researchers:
                if researcher["name"] in paper_authors:
                    collaborative_score += 0.3

            # Check if paper is in areas of interest to similar researchers
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

            optimization_keywords = [
                "adam",
                "sgd",
                "momentum",
                "convergence",
                "gradient",
            ]
            if any(keyword in text for keyword in optimization_keywords):
                collaborative_score += 0.2

            paper["collaborative_score"] = min(collaborative_score, 1.0)

        return papers
