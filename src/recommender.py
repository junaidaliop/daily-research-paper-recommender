import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .embedder import PaperEmbedder
from .paper_fetcher import PaperFetcher
from .profile_builder import ResearchProfileBuilder
from .rag_ranker import RAGRanker

logger = logging.getLogger(__name__)


class ResearchRecommender:
    def __init__(self, config_path="config.yaml"):
        import yaml

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.profile_builder = ResearchProfileBuilder(config_path)
        self.paper_fetcher = PaperFetcher(self.config)
        self.embedder = PaperEmbedder()
        self.rag_ranker = RAGRanker(self.config)

        # Load or build profile
        self.profile = self._load_or_build_profile()

    def _load_or_build_profile(self):
        """Load cached profile or build new one"""
        profile_path = Path("data/profile_cache/profile.pkl")

        if profile_path.exists():
            # Check if profile is recent (less than 7 days old)
            if (
                datetime.now() - datetime.fromtimestamp(profile_path.stat().st_mtime)
            ).days < 7:
                with open(profile_path, "rb") as f:
                    logger.info("Loading cached profile")
                    return pickle.load(f)

        logger.info("Building new profile")
        return self.profile_builder.build_complete_profile()

    def get_daily_recommendations(self) -> List[Dict]:
        """Main method to get daily recommendations"""
        logger.info("Starting daily recommendation process...")

        # 1. Fetch recent papers
        papers = self.paper_fetcher.fetch_daily_papers(days_back=1)
        logger.info(f"Fetched {len(papers)} papers")

        if not papers:
            logger.warning("No papers found")
            return []

        # 2. Create embeddings
        paper_embeddings = self.embedder.embed_papers(papers)

        # 3. Store in vector DB
        self.embedder.store_embeddings(papers, paper_embeddings)

        # 4. Create profile embedding
        profile_embedding = self.embedder.create_profile_embedding(
            self.config["researcher"]
        )

        # 5. Calculate similarities with proper normalization
        # Normalize embeddings
        paper_embeddings_norm = paper_embeddings / np.linalg.norm(
            paper_embeddings, axis=1, keepdims=True
        )
        profile_embedding_norm = profile_embedding / np.linalg.norm(profile_embedding)

        # Calculate cosine similarities (will be between -1 and 1)
        similarities = np.dot(paper_embeddings_norm, profile_embedding_norm)

        # Convert to 0-1 range
        similarities = (similarities + 1) / 2

        # Add similarity scores to papers
        for i, paper in enumerate(papers):
            paper["similarity_score"] = float(similarities[i])
            paper["relevance_score"] = float(similarities[i])  # Default to similarity

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_papers = [papers[i] for i in sorted_indices[:30]]  # Top 30

        # 6. Apply RAG ranking if enabled
        if self.config["recommendations"]["use_rag_reranking"]:
            try:
                ranked_papers = self.rag_ranker.rank_papers(sorted_papers, self.profile)
                # Ensure all papers have relevance scores
                for paper in ranked_papers:
                    if "relevance_score" not in paper or paper["relevance_score"] == 0:
                        paper["relevance_score"] = paper.get("similarity_score", 0.5)
            except Exception as e:
                logger.error(f"RAG ranking failed: {e}")
                ranked_papers = sorted_papers
        else:
            ranked_papers = sorted_papers

        # 7. Ensure all papers have valid relevance scores
        for paper in ranked_papers:
            if "relevance_score" not in paper or paper["relevance_score"] == 0:
                paper["relevance_score"] = paper.get("similarity_score", 0.5)

        # 8. Return top N
        final_recommendations = ranked_papers[
            : self.config["recommendations"]["daily_count"]
        ]

        # 9. Save recommendations
        self._save_recommendations(final_recommendations)

        return final_recommendations

    def _save_recommendations(self, papers: List[Dict]):
        """Save recommendations with timestamp"""
        save_dir = Path("data/recommendations")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as pickle
        with open(save_dir / f"recommendations_{timestamp}.pkl", "wb") as f:
            pickle.dump(papers, f)

        # Save as JSON for easy viewing
        import json

        with open(save_dir / f"recommendations_{timestamp}.json", "w") as f:
            json.dump(papers, f, indent=2, default=str)
