"""
Advanced feedback learning system to improve recommendations over time
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeedbackLearner:
    def __init__(self):
        self.feedback_dir = Path("data/feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.feedback_dir / "feedback_model.pkl"
        self.scaler_path = self.feedback_dir / "scaler.pkl"

        self.feature_extractor = PaperFeatureExtractor()

        # Load or initialize model
        if self.model_path.exists():
            self.load_model()
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False

    def record_feedback(self, paper: Dict, rating: float, action: str = "viewed"):
        """Record user feedback on a paper"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper.get("id", paper["title"]),
            "paper_title": paper["title"],
            "rating": rating,
            "action": action,  # viewed, saved, cited, ignored
            "features": self.feature_extractor.extract_features(paper),
        }

        # Save feedback
        feedback_file = (
            self.feedback_dir / f'feedback_{datetime.now().strftime("%Y%m")}.json'
        )

        import json

        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []

        feedbacks.append(feedback)

        with open(feedback_file, "w") as f:
            json.dump(feedbacks, f, indent=2)

        # Retrain model if enough feedback
        if len(feedbacks) % 20 == 0:
            self.retrain_model()

    def predict_preference(self, paper: Dict) -> float:
        """Predict user preference for a paper"""
        if not self.is_trained:
            return 0.5  # Neutral prediction

        features = self.feature_extractor.extract_features(paper)
        features_scaled = self.scaler.transform([features])

        prediction = self.model.predict(features_scaled)[0]
        return np.clip(prediction, 0, 1)

    def retrain_model(self):
        """Retrain the model with accumulated feedback"""
        logger.info("Retraining feedback model...")

        # Load all feedback
        X, y = [], []

        for feedback_file in self.feedback_dir.glob("feedback_*.json"):
            with open(feedback_file, "r") as f:
                feedbacks = json.load(f)

            for fb in feedbacks:
                X.append(fb["features"])
                y.append(fb["rating"])

        if len(X) < 10:
            logger.warning("Not enough feedback for training")
            return

        # Train model
        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.is_trained = True
        self.save_model()

        logger.info(f"Model retrained with {len(X)} samples")

    def save_model(self):
        """Save trained model and scaler"""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save training metadata
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "is_trained": self.is_trained,
            "feature_importance": (
                dict(
                    zip(
                        self.feature_extractor.feature_names,
                        self.model.feature_importances_,
                    )
                )
                if self.is_trained
                else {}
            ),
        }

        with open(self.feedback_dir / "model_metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

    def load_model(self):
        """Load saved model and scaler"""
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.is_trained = True

    def get_feedback_stats(self) -> Dict:
        """Get statistics about user feedback"""
        all_feedbacks = []

        for feedback_file in self.feedback_dir.glob("feedback_*.json"):
            with open(feedback_file, "r") as f:
                feedbacks = json.load(f)
                all_feedbacks.extend(feedbacks)

        if not all_feedbacks:
            return {}

        df = pd.DataFrame(all_feedbacks)

        stats = {
            "total_feedback": len(df),
            "avg_rating": df["rating"].mean(),
            "rating_distribution": df["rating"].value_counts().to_dict(),
            "action_counts": df["action"].value_counts().to_dict(),
            "recent_feedback": df.tail(10).to_dict("records"),
        }

        return stats


class PaperFeatureExtractor:
    """Extract features from papers for ML models"""

    def __init__(self):
        self.feature_names = [
            "title_length",
            "abstract_length",
            "num_authors",
            "has_optimization",
            "has_fractional",
            "has_neural",
            "has_convergence",
            "has_algorithm",
            "has_deep_learning",
            "num_categories",
            "is_cs_lg",
            "is_math_oc",
            "is_cs_ne",
            "year_published",
            "month_published",
        ]

    def extract_features(self, paper: Dict) -> List[float]:
        """Extract numerical features from a paper"""
        features = []

        # Basic features
        features.append(len(paper.get("title", "").split()))
        features.append(len(paper.get("abstract", "").split()))
        features.append(len(paper.get("authors", [])))

        # Keyword features
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        features.append(1.0 if "optimi" in text else 0.0)
        features.append(1.0 if "fractional" in text else 0.0)
        features.append(1.0 if "neural" in text else 0.0)
        features.append(1.0 if "converge" in text else 0.0)
        features.append(1.0 if "algorithm" in text else 0.0)
        features.append(1.0 if "deep learning" in text else 0.0)

        # Category features
        categories = paper.get("categories", [])
        features.append(len(categories))
        features.append(1.0 if "cs.LG" in categories else 0.0)
        features.append(1.0 if "math.OC" in categories else 0.0)
        features.append(1.0 if "cs.NE" in categories else 0.0)

        # Time features
        if paper.get("published"):
            try:
                pub_date = datetime.fromisoformat(
                    paper["published"].replace("Z", "+00:00")
                )
                features.append(pub_date.year)
                features.append(pub_date.month)
            except:
                features.extend([2025, 1])
        else:
            features.extend([2025, 1])

        return features
