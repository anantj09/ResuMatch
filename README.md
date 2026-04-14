# 📄 ResuMatch — NLP Resume & Job Description Matcher

> NLP Course Project | B.Tech AI&DS  
> Powered by **Sentence-BERT**, **TF-IDF**, and **Streamlit**

---

## 🎯 What it does

ResuMatch analyzes how well a resume matches a job description using:

1. **Semantic Similarity** — SBERT (all-MiniLM-L6-v2) cosine similarity between full texts
2. **Section-wise Breakdown** — Separate similarity scores for Skills, Experience, Education
3. **Keyword Gap Analysis** — TF-IDF extracts keyphrases from JD -> shows matched, missing, and extra keywords
4. **Composite Score** — Weighted combination of the above (50% semantic + 25% keyword + 25% section)
5. **Resume History** — Compare multiple resumes against the same JD with a visual bar chart and exportable CSV

---

## 📁 Directory Structure

```
resume_matcher/
├── app.py                      # Streamlit demo app
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── extractor.py            # PDF / DOCX / TXT text extraction
│   ├── parser.py               # Rule-based section segmentation
│   ├── similarity.py           # SBERT cosine similarity (batched)
│   ├── keywords.py             # TF-IDF keyword extraction + gap analysis
│   └── scorer.py               # Orchestrates all modules → final result dict
│
├── data/
│   └── samples/
│       ├── sample_resume.txt   # Sample NLP engineer resume
│       └── sample_jd.txt       # Sample NLP engineer JD
│
├── notebooks/
│   └── exploration.ipynb       # EDA and model experiments
│
└── tests/
    └── test_core.py            # pytest unit tests
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
git clone <repo-url>
cd resume_matcher
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Quick test:** Click "Load sample resume & JD" inside the app, then hit "Analyze Match".

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🧠 Model Details

| Component | Model / Method |
|---|---|
| Semantic similarity | `all-MiniLM-L6-v2` (SBERT) |
| Keyword extraction | TF-IDF bigrams (scikit-learn) |
| Section parsing | Rule-based regex header matching |
| File parsing | PyMuPDF (PDF), python-docx (DOCX) |

---

## 📊 Scoring Formula

```
Overall Score = 0.50 × Semantic Similarity
              + 0.25 × Keyword Match Rate
              + 0.25 × Average Section Similarity
```

Score is multiplied by 100 and displayed as a percentage.

| Score Range | Verdict      |
|-------------|--------------|
| 75 – 100    | Strong Match |
| 55 – 74     | Good Match   |
| 35 – 54     | Fair Match   |
| 0 – 34      | Weak Match   |

---

## 📦 Key Dependencies

- `sentence-transformers` — SBERT embeddings
- `streamlit` — Web demo
- `plotly` — Interactive charts
- `PyMuPDF` — PDF parsing
- `python-docx` — DOCX parsing
- `scikit-learn` — TF-IDF keyword extraction, vectorization

---

## 👤 Author

Anant Jain