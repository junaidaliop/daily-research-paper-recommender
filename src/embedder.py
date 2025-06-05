import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PaperEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB
        db_path = Path("data/embeddings_db")
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"description": "Research paper embeddings"},
        )

    def embed_papers(self, papers: List[Dict]) -> np.ndarray:
        """Create embeddings for papers"""
        texts = []
        for paper in papers:
            # Combine title and abstract for richer embedding
            text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            texts.append(text)

        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def store_embeddings(self, papers: List[Dict], embeddings: np.ndarray):
        """Store paper embeddings in ChromaDB"""
        ids = []
        documents = []
        metadatas = []

        for i, paper in enumerate(papers):
            # Create unique ID
            paper_id = hashlib.md5(paper["title"].encode()).hexdigest()
            ids.append(paper_id)

            # Document text
            doc_text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            documents.append(doc_text)

            # Metadata
            metadata = {
                "title": paper["title"],
                "authors": ", ".join(paper["authors"][:3]),  # First 3 authors
                "published": paper.get("published", ""),
                "source": paper.get("source", ""),
                "url": paper.get("pdf_url", paper.get("url", "")),
                "categories": ", ".join(paper.get("categories", [])),
            }
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(papers)} papers in vector DB")

    def search_similar(
        self, query_embedding: np.ndarray, n_results: int = 20
    ) -> List[Dict]:
        """Search for similar papers"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results
        )

        papers = []
        for i in range(len(results["ids"][0])):
            paper = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "metadata": results["metadatas"][0][i],
            }
            papers.append(paper)

        return papers

    def create_profile_embedding(self, profile_data: Dict) -> np.ndarray:
        """Create embedding from user profile"""
        profile_texts = []

        # Add research interests with higher weight
        if "primary_interests" in profile_data:
            # Repeat primary interests to give them more weight
            for interest in profile_data["primary_interests"]:
                profile_texts.extend([interest] * 3)  # Triple weight

        # Add keywords
        if "keywords" in profile_data:
            for category, keywords in profile_data["keywords"].items():
                profile_texts.extend(keywords)

        # Add favorite paper titles/topics
        favorite_topics = [
            "fractional order optimization",
            "adaptive learning rates",
            "neural differential equations",
            "convergence analysis",
            "gradient descent variants",
        ]
        profile_texts.extend(favorite_topics)

        # Create combined embedding
        if profile_texts:
            embeddings = self.model.encode(profile_texts)
            # Weighted average
            return np.mean(embeddings, axis=0)
        else:
            # Fallback
            return self.model.encode(
                ["optimization algorithms deep learning fractional calculus"]
            )[0]
