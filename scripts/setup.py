#!/usr/bin/env python3
"""
Setup script to initialize the recommendation system
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from pathlib import Path

import yaml


def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/profile_cache",
        "data/embeddings_db",
        "data/recommendations",
        "data/recommendations/reports",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✓ Created directory structure")


def install_ollama_model():
    """Pull the Ollama model"""
    try:
        print("Pulling Mixtral model (this may take a while)...")
        subprocess.run(
            ["ollama", "pull", "mixtral:8x7b-instruct-v0.1-q4_0"], check=True
        )
        print("✓ Ollama model ready")
    except:
        print("⚠️  Could not pull Ollama model. Make sure Ollama is installed.")
        print("   Install from: https://ollama.ai")


def validate_config():
    """Validate configuration"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("\nConfiguration Summary:")
    print(f"- Researcher: {config['researcher']['name']}")
    print(f"- Google Scholar ID: {config['researcher']['google_scholar_id']}")
    print(f"- GitHub: {config['researcher']['github_username']}")
    print(f"- LLM Provider: {config['llm']['provider']}")
    print(f"- Daily recommendations: {config['recommendations']['daily_count']}")

    if config["llm"]["provider"] == "groq" and not config["llm"]["groq_api_key"]:
        print("\n⚠️  Groq selected but no API key provided.")
        print("   Get free key at: https://console.groq.com")


def setup_cron():
    """Setup daily cron job"""
    print("\nTo run daily, add this to your crontab (crontab -e):")
    print(
        "0 9 * * * cd //home/efreet/Ali/research-recommender && python scripts/run_daily.py"
    )


def main():
    print("Setting up Research Recommender System...")

    setup_directories()
    validate_config()
    install_ollama_model()
    setup_cron()

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit config.yaml with your email settings (optional)")
    print("2. Run: python scripts/run_daily.py")
    print("3. Check data/recommendations/reports/ for your recommendations")


if __name__ == "__main__":
    main()
