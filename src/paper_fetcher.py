import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List

import arxiv
import feedparser
import requests

logger = logging.getLogger(__name__)


class PaperFetcher:
    def __init__(self, config):
        self.config = config
        self.arxiv_client = arxiv.Client()

    def fetch_daily_papers(self, days_back=1) -> List[Dict]:
        """Fetch papers from multiple sources"""
        all_papers = []

        # Fetch from arXiv
        arxiv_papers = self._fetch_arxiv_papers(days_back)
        all_papers.extend(arxiv_papers)

        # Fetch from Semantic Scholar
        s2_papers = self._fetch_semantic_scholar_papers(days_back)
        all_papers.extend(s2_papers)

        # Remove duplicates
        seen = set()
        unique_papers = []
        for paper in all_papers:
            if paper["title"] not in seen:
                seen.add(paper["title"])
                unique_papers.append(paper)

        logger.info(f"Fetched {len(unique_papers)} unique papers")
        return unique_papers

    def _fetch_arxiv_papers(self, days_back=1) -> List[Dict]:
        """Fetch recent papers from arXiv"""
        papers = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for category in self.config["sources"]["arxiv"]["categories"]:
            try:
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=self.config["sources"]["arxiv"][
                        "max_results_per_category"
                    ],
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )

                for result in self.arxiv_client.results(search):
                    if result.published.replace(tzinfo=None) >= cutoff_date:
                        papers.append(
                            {
                                "id": result.entry_id,
                                "title": result.title,
                                "abstract": result.summary,
                                "authors": [a.name for a in result.authors],
                                "categories": result.categories,
                                "published": result.published.isoformat(),
                                "pdf_url": result.pdf_url,
                                "source": "arxiv",
                            }
                        )

                time.sleep(0.5)  # Be respectful

            except Exception as e:
                logger.error(f"Error fetching from arXiv {category}: {e}")

        return papers

    def _fetch_semantic_scholar_papers(self, days_back=1) -> List[Dict]:
        """Fetch papers from Semantic Scholar"""
        papers = []

        # Search for papers related to our interests
        for interest in self.config["researcher"]["primary_interests"][
            :3
        ]:  # Top 3 interests
            try:
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": interest,
                    "fields": "title,abstract,authors,year,publicationDate,url",
                    "limit": 20,
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()

                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    for paper in data.get("data", []):
                        if paper.get("publicationDate"):
                            pub_date = datetime.fromisoformat(
                                paper["publicationDate"].replace("Z", "+00:00")
                            )
                            if pub_date.replace(tzinfo=None) >= cutoff_date:
                                papers.append(
                                    {
                                        "id": paper.get("paperId", ""),
                                        "title": paper.get("title", ""),
                                        "abstract": paper.get("abstract", ""),
                                        "authors": [
                                            a["name"] for a in paper.get("authors", [])
                                        ],
                                        "published": paper.get("publicationDate", ""),
                                        "url": paper.get("url", ""),
                                        "source": "semantic_scholar",
                                    }
                                )

                time.sleep(1)  # Rate limit

            except Exception as e:
                logger.error(f"Error fetching from Semantic Scholar: {e}")

        return papers

    def fetch_specific_papers(self, paper_ids: List[str]) -> List[Dict]:
        """Fetch specific papers by ID (for analyzing favorites)"""
        papers = []

        for paper_id in paper_ids:
            if "arxiv" in paper_id:
                # Extract arXiv ID
                arxiv_id = paper_id.split("/")[-1]
                try:
                    search = arxiv.Search(id_list=[arxiv_id])
                    result = next(self.arxiv_client.results(search))
                    papers.append(
                        {
                            "id": result.entry_id,
                            "title": result.title,
                            "abstract": result.summary,
                            "authors": [a.name for a in result.authors],
                            "categories": result.categories,
                        }
                    )
                except:
                    logger.error(f"Could not fetch paper {arxiv_id}")

        return papers
