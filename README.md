# 🚀 AI Research Paper Recommender

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.ai)
[![arXiv](https://img.shields.io/badge/arXiv-API-b31b1b.svg)](https://arxiv.org)

**Get personalized research paper recommendations delivered daily, powered by AI that understands YOUR research interests!**

Never miss important papers in your field again. This intelligent system analyzes your Google Scholar profile, GitHub stars, and research interests to find the most relevant new papers from arXiv, Semantic Scholar, and more.

## ✨ Features

- 🎯 **Personalized Recommendations** - Analyzes your publications and interests
- 🤖 **AI-Powered Ranking** - Uses RAG (Retrieval-Augmented Generation) to explain WHY each paper matters to YOU
- 📊 **Beautiful Dashboard** - Interactive Streamlit web interface with dark mode
- 📧 **Daily Email Digest** - Get recommendations delivered to your inbox
- 🔄 **Continuous Learning** - Rate papers to improve future recommendations
- 🆓 **100% Free** - Runs locally with Ollama, no API costs

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/junaidaliop/daily-research-paper-recommender.git
cd daily-research-paper-recommender

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama (for AI)
curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac
# Windows: Download from https://ollama.com

# Run setup
python scripts/setup.py

# Get recommendations
python scripts/run_daily.py

# Or use web interface
streamlit run streamlit_app.py
```

## ⚙️ Configuration

### 1. Research Profile

Edit `config.yaml`:

```yaml
researcher:
  name: "Your Name"
  google_scholar_id: "YOUR_ID"
  primary_interests:
    - "your research area"
    - "your interests"
```

## 📚 How It Works

1. **Fetches** papers from arXiv & Semantic Scholar
2. **Analyzes** with semantic embeddings (BGE-M3)
3. **Ranks** using AI to explain relevance
4. **Delivers** via web interface or email

## 🛠️ Tech Stack

- **AI**: Sentence-Transformers, Ollama/Groq
- **Data**: ChromaDB, NumPy, scikit-learn
- **Web**: Streamlit, Plotly
- **Sources**: arXiv, Semantic Scholar APIs

## 🏗️ Project Structure

```
daily-research-paper-recommender/
├── src/
│   ├── profile_builder.py    # Analyzes your research profile
│   ├── paper_fetcher.py      # Fetches papers from sources
│   ├── embedder.py           # Creates semantic embeddings
│   ├── recommender.py        # Main recommendation engine
│   ├── rag_ranker.py         # AI-powered ranking
│   └── notifier.py           # Sends notifications
├── scripts/
│   ├── run_daily.py          # Daily automation
│   └── setup.py              # Initial setup
├── streamlit_app.py          # Web interface
└── config.yaml               # Configuration
```

## 🤝 Contributing

We love contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation
- 🎨 UI improvements

## 📧 Contact

**Muhammad Junaid Ali**  
Email: <muhammadjunaidaliasifraja@gmail.com>  
GitHub: [@junaidaliop](https://github.com/junaidaliop)

## 🙏 Acknowledgments

- [arXiv](https://arxiv.org) for providing open access to papers
- [Semantic Scholar](https://www.semanticscholar.org) for their API
- [Ollama](https://ollama.ai) for local LLM inference
- [Streamlit](https://streamlit.io) for the amazing web framework

---

<p align="center">
  Made with ❤️ by Junaid Ali, for researchers
  <br>
  ⭐ Star us on GitHub if this helps your research!
</p>

---

**Keywords**: research paper recommender, academic paper recommendation, arXiv papers, machine learning papers, AI research assistant, scholarly recommendations, personalized paper discovery, research automation, academic AI tools, paper recommendation system

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---
