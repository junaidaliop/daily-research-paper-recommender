import json
import logging
import os
from typing import Dict, List

import ollama
from groq import Groq

logger = logging.getLogger(__name__)


class RAGRanker:
    def __init__(self, config):
        self.config = config

        if config["llm"]["provider"] == "groq" and config["llm"]["groq_api_key"]:
            self.client = Groq(api_key=config["llm"]["groq_api_key"])
            self.use_groq = True
        else:
            self.client = ollama.Client()
            self.use_groq = False
            # Pull model if not exists
            try:
                ollama.pull(config["llm"]["model"])
            except:
                pass

    def rank_papers(self, papers: List[Dict], profile: Dict) -> List[Dict]:
        """Use LLM to rank and explain paper relevance"""

        # Create research profile summary
        profile_summary = self._create_profile_summary(profile)

        # Prepare papers for ranking
        papers_text = self._format_papers_for_llm(papers[:15])  # Top 15 candidates

        prompt = f"""You are an expert in optimization algorithms, fractional calculus, and computational neuroscience.

    Research Profile:
    {profile_summary}

    Task: Rank these papers by relevance to the researcher's interests and explain why each is relevant.

    Papers to rank:
    {papers_text}

    Return a JSON array with this EXACT structure (make sure relevance_score is a number between 0 and 1):
    [
    {{
        "rank": 1,
        "title": "exact paper title here",
        "relevance_score": 0.95,
        "explanation": "This paper introduces a fractional-order Adam optimizer...",
        "key_contributions": ["contribution 1", "contribution 2"],
        "potential_applications": "Could be applied to your work on..."
    }}
    ]

    Important: relevance_score MUST be a decimal number between 0 and 1. Only include papers with relevance_score > 0.7."""

        try:
            if self.use_groq:
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content
            else:
                response = self.client.generate(
                    model=self.config["llm"]["model"], prompt=prompt, stream=False
                )
                content = response["response"]

            # Parse JSON response
            import re

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                content = json_match.group()

            ranked_papers = json.loads(content)

            # Match back to original papers and ensure scores are floats
            result = []
            for ranked in ranked_papers:
                for paper in papers:
                    if (
                        ranked["title"].lower() in paper["title"].lower()
                        or paper["title"].lower() in ranked["title"].lower()
                    ):
                        paper["rank"] = ranked["rank"]
                        # Ensure relevance_score is a valid float
                        try:
                            paper["relevance_score"] = float(
                                ranked.get("relevance_score", 0.8)
                            )
                        except:
                            paper["relevance_score"] = 0.8
                        paper["explanation"] = ranked.get(
                            "explanation", "Relevant to your research interests."
                        )
                        paper["key_contributions"] = ranked.get("key_contributions", [])
                        paper["potential_applications"] = ranked.get(
                            "potential_applications", ""
                        )
                        result.append(paper)
                        break

            # If no papers matched, return original with similarity scores
            if not result:
                logger.warning(
                    "No papers matched in RAG ranking, using similarity scores"
                )
                for i, paper in enumerate(papers[:10]):
                    paper["rank"] = i + 1
                    paper["relevance_score"] = paper.get("similarity_score", 0.7)
                    paper["explanation"] = (
                        f"Relevant based on similarity to your research profile (score: {paper['relevance_score']:.2f})"
                    )
                result = papers[:10]

            return sorted(result, key=lambda x: x.get("rank", 999))

        except Exception as e:
            logger.error(f"Error in LLM ranking: {e}")
            # Fallback: use similarity scores
            for i, paper in enumerate(papers[:10]):
                paper["rank"] = i + 1
                paper["relevance_score"] = paper.get("similarity_score", 0.7)
                paper["explanation"] = (
                    "Relevant to your research interests based on content similarity."
                )
            return papers[:10]

    def _create_profile_summary(self, profile: Dict) -> str:
        """Create a summary of the research profile"""
        summary = "Research Focus:\n"

        # Add primary interests
        interests = profile.get("primary_interests", [])
        summary += f"- Primary interests: {', '.join(interests)}\n"

        # Add recent papers
        if "scholar_data" in profile and profile["scholar_data"]:
            recent_papers = profile["scholar_data"].get("papers", [])[:5]
            if recent_papers:
                summary += "\nRecent publications:\n"
                for paper in recent_papers:
                    summary += f"- {paper['title']} ({paper.get('year', 'N/A')})\n"

        # Add favorite paper patterns
        summary += (
            "\nFavorite paper themes: fractional-order methods, novel optimizers, "
        )
        summary += (
            "neural ODEs, computational efficiency, theoretical convergence analysis\n"
        )

        return summary

    def _format_papers_for_llm(self, papers: List[Dict]) -> str:
        """Format papers for LLM processing"""
        formatted = []
        for i, paper in enumerate(papers):
            text = f"\n{i+1}. Title: {paper['title']}\n"
            text += f"   Abstract: {paper['abstract'][:300]}...\n"
            text += f"   Categories: {', '.join(paper.get('categories', []))}\n"
            formatted.append(text)

        return "\n".join(formatted)

    def generate_summary_report(self, papers: List[Dict]) -> str:
        """Generate a natural language summary of recommendations"""
        prompt = f"""Write a brief, engaging summary of today's paper recommendations for a researcher
        working on optimization algorithms and fractional calculus.

        Mention key themes, breakthroughs, and why these papers matter.
        Keep it under 200 words and conversational.

        Papers: {[p['title'] for p in papers[:5]]}"""

        try:
            if self.use_groq:
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            else:
                response = self.client.generate(
                    model=self.config["llm"]["model"], prompt=prompt
                )
                return response["response"]
        except:
            return "Today's recommendations focus on advances in optimization and fractional calculus."
