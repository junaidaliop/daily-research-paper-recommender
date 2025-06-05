#!/usr/bin/env python3
"""
Daily recommendation script
Run this with cron or task scheduler
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import yaml

from src.notifier import RecommendationNotifier
from src.rag_ranker import RAGRanker
from src.recommender import ResearchRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/recommendations.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting daily recommendation run...")

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        # Initialize recommender
        recommender = ResearchRecommender("config.yaml")

        # Get recommendations
        recommendations = recommender.get_daily_recommendations()

        if recommendations:
            logger.info(f"Generated {len(recommendations)} recommendations")

            # Generate summary
            ranker = RAGRanker(config)
            summary = ranker.generate_summary_report(recommendations)

            # Send notifications
            notifier = RecommendationNotifier(config)
            notifier.send_notifications(recommendations, summary)

            logger.info("Daily recommendations completed successfully")
        else:
            logger.warning("No recommendations generated")

    except Exception as e:
        logger.error(f"Error in daily run: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
