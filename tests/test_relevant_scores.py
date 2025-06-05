import os

# Add parent directory to path so we can import src modules
import os.path
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import ResearchRecommender


def test_relevance_scores():
    print("Testing relevance score calculation...")

    # Initialize recommender
    recommender = ResearchRecommender("config.yaml")

    # Get recommendations
    recs = recommender.get_daily_recommendations()

    print(f"\nGot {len(recs)} recommendations:")
    for i, paper in enumerate(recs[:5]):
        print(f"\n{i+1}. {paper['title'][:60]}...")
        print(f"   Relevance Score: {paper.get('relevance_score', 0):.3f}")
        print(f"   Similarity Score: {paper.get('similarity_score', 0):.3f}")
        if "explanation" in paper:
            print(f"   Explanation: {paper['explanation'][:100]}...")


if __name__ == "__main__":
    test_relevance_scores()
