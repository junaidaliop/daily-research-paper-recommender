"""
Predict future citation impact of papers
"""

import re
from datetime import datetime
from typing import Dict

import numpy as np


class CitationPredictor:
    """Predict which papers might become highly cited"""

    def __init__(self):
        # Factors that correlate with high citations
        self.high_impact_venues = [
            "neurips",
            "icml",
            "iclr",
            "cvpr",
            "nature",
            "science",
        ]
        self.high_impact_authors = ["bengio", "lecun", "hinton", "goodfellow", "sutton"]
        self.trending_keywords = [
            "transformer",
            "diffusion",
            "llm",
            "foundation model",
            "fractional",
            "neural ode",
            "optimization landscape",
        ]

    def predict_impact(self, paper: Dict) -> Dict[str, float]:
        """Predict citation impact of a paper"""
        scores = {
            "venue_score": 0.0,
            "author_score": 0.0,
            "novelty_score": 0.0,
            "trend_score": 0.0,
            "overall_score": 0.0,
        }

        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        authors_text = " ".join(paper.get("authors", [])).lower()

        # Venue impact
        for venue in self.high_impact_venues:
            if venue in text:
                scores["venue_score"] = 0.8
                break

        # Author impact
        for author in self.high_impact_authors:
            if author in authors_text:
                scores["author_score"] = 0.9
                break

        # Novelty indicators
        novelty_patterns = [
            r"first\s+\w+\s+to",
            r"novel\s+\w+",
            r"new\s+\w+\s+for",
            r"introduce\s+\w+",
            r"propose\s+\w+",
        ]

        novelty_count = sum(
            1 for pattern in novelty_patterns if re.search(pattern, text)
        )
        scores["novelty_score"] = min(novelty_count * 0.2, 1.0)

        # Trending topics
        trend_count = sum(1 for keyword in self.trending_keywords if keyword in text)
        scores["trend_score"] = min(trend_count * 0.3, 1.0)

        # Overall score
        scores["overall_score"] = np.mean(
            [
                scores["venue_score"],
                scores["author_score"],
                scores["novelty_score"],
                scores["trend_score"],
            ]
        )

        return scores


# src/research_gap_analyzer.py
"""
Analyze research gaps and suggest new directions
"""

from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx


class ResearchGapAnalyzer:
    """Identify research gaps and opportunities"""

    def __init__(self, user_papers: List[Dict]):
        self.user_papers = user_papers
        self.research_graph = self._build_research_graph()

    def _build_research_graph(self) -> nx.Graph:
        """Build a graph of research topics and their connections"""
        G = nx.Graph()

        # Extract topics from user's papers
        for paper in self.user_papers:
            title = paper.get("title", "").lower()

            # Simple keyword extraction
            keywords = []
            if "optimization" in title:
                keywords.append("optimization")
            if "fractional" in title:
                keywords.append("fractional")
            if "neural" in title:
                keywords.append("neural")
            if "convergence" in title:
                keywords.append("convergence")

            # Add edges between co-occurring topics
            for i, kw1 in enumerate(keywords):
                G.add_node(kw1)
                for kw2 in keywords[i + 1 :]:
                    G.add_edge(kw1, kw2)

        return G

    def identify_gaps(self, new_papers: List[Dict]) -> List[Dict]:
        """Identify papers that bridge research gaps"""
        gap_papers = []

        for paper in new_papers:
            gap_score = 0.0
            reasons = []

            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            text = f"{title} {abstract}"

            # Check for bridging topics
            bridges = [
                (
                    "fractional",
                    "neural",
                    "Bridges fractional calculus and neural networks",
                ),
                (
                    "optimization",
                    "fractional",
                    "Connects optimization with fractional methods",
                ),
                (
                    "convergence",
                    "fractional",
                    "Analyzes convergence in fractional systems",
                ),
                ("neural", "ode", "Neural ODEs - continuous neural models"),
            ]

            for topic1, topic2, reason in bridges:
                if topic1 in text and topic2 in text:
                    gap_score += 0.5
                    reasons.append(reason)

            # Check for novel combinations
            novel_combinations = [
                (["fractional", "adam"], "Fractional-order Adam optimizer"),
                (
                    ["neural", "fractional", "ode"],
                    "Neural fractional differential equations",
                ),
                (
                    ["optimization", "landscape", "fractional"],
                    "Fractional optimization landscapes",
                ),
            ]

            for keywords, reason in novel_combinations:
                if all(kw in text for kw in keywords):
                    gap_score += 0.7
                    reasons.append(reason)

            if gap_score > 0:
                paper["gap_score"] = min(gap_score, 1.0)
                paper["gap_reasons"] = reasons
                gap_papers.append(paper)

        return sorted(gap_papers, key=lambda x: x["gap_score"], reverse=True)
