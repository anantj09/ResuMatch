"""
src/scorer.py
-------------
Orchestrates similarity scoring, section breakdown, and keyword gap
into a single unified result dict consumed by the Streamlit app.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.similarity import compute_similarity, compute_section_similarities
from src.parser import parse_sections
from src.keywords import keyword_gap_analysis

# Weights for the composite overall score
_WEIGHTS = {
    "semantic_overall": 0.50,
    "keyword_match_rate": 0.25,
    "section_avg": 0.25,
}


def score(resume_text: str, jd_text: str, model: SentenceTransformer) -> dict:
    """
    Run full analysis pipeline.

    Returns a result dict with keys:
        overall_score       : float [0, 100]
        semantic_score      : float [0, 1]
        keyword_gap         : dict (from keywords.py)
        section_scores      : dict {skills/experience/education: float|None}
        resume_sections     : dict of parsed resume text
        jd_sections         : dict of parsed JD text
    """
    import time

    t = time.time()
    semantic = compute_similarity(model, resume_text, jd_text)
    print(f"  [scorer] similarity: {time.time()-t:.2f}s")

    t = time.time()
    resume_secs = parse_sections(resume_text)
    jd_secs = parse_sections(jd_text)
    print(f"  [scorer] parsing: {time.time()-t:.2f}s")

    t = time.time()
    section_scores = compute_section_similarities(model, resume_secs, jd_secs)
    print(f"  [scorer] sections: {time.time()-t:.2f}s")

    t = time.time()
    kw_gap = keyword_gap_analysis(resume_text, jd_text, model=model)
    print(f"  [scorer] keywords: {time.time()-t:.2f}s")

    valid_section_scores = [v for v in section_scores.values() if v is not None]
    section_avg = sum(valid_section_scores) / max(len(valid_section_scores), 1)
    composite = (
        _WEIGHTS["semantic_overall"] * semantic
        + _WEIGHTS["keyword_match_rate"] * kw_gap["match_rate"]
        + _WEIGHTS["section_avg"] * section_avg
    )

    return {
        "overall_score": round(composite * 100, 1),
        "semantic_score": semantic,
        "keyword_gap": kw_gap,
        "section_scores": section_scores,
        "resume_sections": resume_secs,
        "jd_sections": jd_secs,
    }