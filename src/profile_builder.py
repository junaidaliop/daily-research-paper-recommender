import logging
import pickle
from pathlib import Path

import numpy as np
import requests
import scholarly
import yaml
from github import Github
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchProfileBuilder:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.embedder = SentenceTransformer("BAAI/bge-m3")
        self.profile_cache_dir = Path("data/profile_cache")
        self.profile_cache_dir.mkdir(parents=True, exist_ok=True)

    def build_complete_profile(self):
        """Build comprehensive research profile"""
        profile = {
            "scholar_data": self._fetch_scholar_profile(),
            "github_interests": self._analyze_github_stars(),
            "paper_embeddings": self._create_paper_embeddings(),
            "interest_embeddings": self._create_interest_embeddings(),
            "favorite_papers": self._analyze_favorite_papers(),
        }

        # Save profile
        with open(self.profile_cache_dir / "profile.pkl", "wb") as f:
            pickle.dump(profile, f)

        return profile

    def _fetch_scholar_profile(self):
        """Fetch and analyze Google Scholar profile"""
        logger.info("Fetching Google Scholar profile...")

        try:
            # Updated API - search_author_id was removed in newer versions
            author_id = self.config["researcher"]["google_scholar_id"]
            author = (
                scholarly.search_author_id(author_id)
                if hasattr(scholarly, "search_author_id")
                else None
            )

            if not author:
                # Alternative method using direct URL
                from scholarly import ProxyGenerator

                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)

                # Search by name as fallback
                search_query = scholarly.search_author(
                    self.config["researcher"]["name"]
                )
                author = next(search_query, None)

            if author:
                author = scholarly.fill(author)

            papers_data = []
            for pub in author.get("publications", [])[:30]:  # Last 30 papers
                try:
                    pub_filled = scholarly.fill(pub)
                    papers_data.append(
                        {
                            "title": pub_filled.get("bib", {}).get("title", ""),
                            "abstract": pub_filled.get("bib", {}).get("abstract", ""),
                            "year": pub_filled.get("bib", {}).get("pub_year", ""),
                            "citations": pub_filled.get("num_citations", 0),
                        }
                    )
                except:
                    continue

            return {
                "name": author.get("name"),
                "papers": papers_data,
                "interests": author.get("interests", []),
                "h_index": author.get("hindex", 0),
                "total_citations": author.get("citedby", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching Scholar profile: {e}")
            return None

    def _analyze_github_stars(self):
        """Analyze GitHub starred repositories"""
        logger.info("Analyzing GitHub stars...")

        try:
            g = Github()  # No token needed for public data
            user = g.get_user(self.config["researcher"]["github_username"])

            starred_topics = []
            for repo in user.get_starred()[:50]:  # Last 50 stars
                starred_topics.extend(repo.topics)
                if repo.description:
                    starred_topics.append(repo.description)

            return starred_topics
        except Exception as e:
            logger.error(f"Error analyzing GitHub: {e}")
            return []

    def _create_paper_embeddings(self):
        """Create embeddings from published papers"""
        scholar_data = self._fetch_scholar_profile()
        if not scholar_data:
            return None

        paper_texts = []
        for paper in scholar_data["papers"]:
            text = f"{paper['title']} {paper['abstract']}"
            if text.strip():
                paper_texts.append(text)

        if paper_texts:
            embeddings = self.embedder.encode(paper_texts)
            return np.mean(embeddings, axis=0)  # Average embedding
        return None

    def _create_interest_embeddings(self):
        """Create embeddings from research interests"""
        interests = []

        # From config
        interests.extend(self.config["researcher"]["primary_interests"])

        # From keywords
        for category, keywords in self.config["researcher"]["keywords"].items():
            interests.extend(keywords)

        if interests:
            embeddings = self.embedder.encode(interests)
            return embeddings
        return None

    def _analyze_favorite_papers(self):
        """Analyze patterns in favorite papers"""
        favorite_papers = [
            "https://arxiv.org/abs/2208.06677",  # Your provided favorites
            "https://arxiv.org/abs/2412.13148",
            "https://arxiv.org/abs/2411.02853",
            "https://arxiv.org/abs/2412.05270",
            "https://arxiv.org/abs/2403.03507",
        ]

        # We'll fetch these papers and analyze them
        # Implementation in paper_fetcher.py
        return favorite_papers
        return favorite_papers
