"""
Analyze user's favorite papers to understand preferences
"""

import os
import sys

# Add parent directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arxiv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import PaperEmbedder


def analyze_favorite_papers():
    # Your favorite papers
    favorite_ids = [
        "2208.06677",  # Likely about optimization
        "2412.13148",  # Recent paper
        "2411.02853",  # Recent paper
        "2412.05270",  # Very recent
        "2403.03507",  # From earlier this year
    ]

    # Fetch papers
    papers = []
    for paper_id in favorite_ids:
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(arxiv.Client().results(search))
            papers.append(
                {
                    "id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "categories": result.categories,
                }
            )
            print(f"✓ Fetched: {result.title}")
        except Exception as e:
            print(f"✗ Error fetching {paper_id}: {e}")

    # Analyze patterns
    print("\n=== ANALYSIS ===")

    # Categories
    all_categories = []
    for paper in papers:
        all_categories.extend(paper["categories"])

    print("\nCategories:")
    from collections import Counter

    cat_counts = Counter(all_categories)
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Keywords
    print("\nCommon keywords:")
    all_text = " ".join([p["title"] + " " + p["abstract"] for p in papers]).lower()

    keywords = [
        "optimization",
        "gradient",
        "convergence",
        "neural",
        "fractional",
        "algorithm",
        "adaptive",
        "learning rate",
        "momentum",
        "descent",
    ]

    for keyword in keywords:
        count = all_text.count(keyword)
        if count > 0:
            print(f"  {keyword}: {count}")

    # Embeddings similarity
    embedder = PaperEmbedder()
    embeddings = embedder.embed_papers(papers)

    # Calculate pairwise similarities
    print("\nPaper similarities:")
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"  {i+1} vs {j+1}: {sim:.3f}")

    # Visualize
    if len(papers) > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

        for i, paper in enumerate(papers):
            plt.annotate(
                f"{i+1}: {paper['title'][:30]}...",
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
            )

        plt.title("Favorite Papers Embedding Space")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.tight_layout()
        plt.savefig("favorite_papers_analysis.png")
        print("\n✓ Saved visualization to favorite_papers_analysis.png")


if __name__ == "__main__":
    analyze_favorite_papers()
