import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class RecommendationNotifier:
    def __init__(self, config):
        self.config = config

    def send_notifications(self, papers: List[Dict], summary: str = ""):
        """Send recommendations via configured channels"""

        if self.config["notifications"]["markdown_report"]:
            self._create_markdown_report(papers, summary)

        if (
            self.config["notifications"]["enabled"]
            and self.config["notifications"]["email"]["sender"]
        ):
            self._send_email(papers, summary)

    def _create_markdown_report(self, papers: List[Dict], summary: str):
        """Create a markdown report"""
        report_dir = Path("data/recommendations/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = report_dir / f"daily_recommendations_{date_str}.md"

        with open(report_path, "w") as f:
            f.write(f"# Research Paper Recommendations - {date_str}\n\n")

            if summary:
                f.write(f"## Summary\n\n{summary}\n\n")

            f.write("## Recommended Papers\n\n")

            for i, paper in enumerate(papers, 1):
                f.write(f"### {i}. {paper['title']}\n\n")

                f.write(f"**Authors**: {', '.join(paper['authors'][:3])}")
                if len(paper["authors"]) > 3:
                    f.write(" et al.")
                f.write("\n\n")

                if "relevance_score" in paper:
                    f.write(f"**Relevance Score**: {paper['relevance_score']:.2f}\n\n")

                if "explanation" in paper:
                    f.write(f"**Why Relevant**: {paper['explanation']}\n\n")

                if "key_contributions" in paper and paper["key_contributions"]:
                    f.write("**Key Contributions**:\n")
                    for contrib in paper["key_contributions"]:
                        f.write(f"- {contrib}\n")
                    f.write("\n")

                f.write(f"**Abstract**: {paper['abstract'][:500]}...\n\n")

                if paper.get("pdf_url"):
                    f.write(f"**Paper**: [{paper['pdf_url']}]({paper['pdf_url']})\n\n")
                elif paper.get("url"):
                    f.write(f"**Paper**: [{paper['url']}]({paper['url']})\n\n")

                f.write("---\n\n")

        logger.info(f"Markdown report saved to {report_path}")
        return report_path

    def _send_email(self, papers: List[Dict], summary: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"Daily Research Papers - {datetime.now().strftime('%Y-%m-%d')}"
            )
            msg["From"] = self.config["notifications"]["email"]["sender"]
            msg["To"] = self.config["notifications"]["email"]["recipient"]

            # Create HTML content
            html_content = self._create_html_email(papers, summary)

            # Attach HTML
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            with smtplib.SMTP(
                self.config["notifications"]["email"]["smtp_server"],
                self.config["notifications"]["email"]["smtp_port"],
            ) as server:
                server.starttls()
                server.login(
                    self.config["notifications"]["email"]["sender"],
                    self.config["notifications"]["email"]["password"],
                )
                server.send_message(msg)

            logger.info("Email notification sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _create_html_email(self, papers: List[Dict], summary: str) -> str:
        """Create HTML email content"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .paper {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f4f4f4;
                    border-radius: 8px;
                }}
                .title {{ color: #333; font-size: 18px; font-weight: bold; }}
                .relevance {{ color: #007bff; font-weight: bold; }}
                .explanation {{
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .abstract {{ color: #666; }}
                a {{ color: #007bff; text-decoration: none; }}
            </style>
        </head>
        <body>
            <h2>Daily Research Paper Recommendations</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

            {f'<div class="summary"><h3>Summary</h3><p>{summary}</p></div>' if summary else ''}

            <h3>Recommended Papers</h3>
        """

        for i, paper in enumerate(papers, 1):
            html += f"""
            <div class="paper">
                <div class="title">{i}. {paper['title']}</div>
                <p><strong>Authors:</strong> {', '.join(paper['authors'][:3])}
                   {' et al.' if len(paper['authors']) > 3 else ''}</p>

                {f'<p class="relevance">Relevance Score: {paper.get("relevance_score", 0):.2f}</p>'
                 if 'relevance_score' in paper else ''}

                {f'<div class="explanation"><strong>Why Relevant:</strong> {paper["explanation"]}</div>'
                 if 'explanation' in paper else ''}

                <p class="abstract"><strong>Abstract:</strong> {paper['abstract'][:400]}...</p>

                <p><a href="{paper.get('pdf_url', paper.get('url', '#'))}">Read Paper →</a></p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html
