"""
src/similarity.py
-----------------
Semantic similarity scoring between resume and job description
using Sentence-BERT (all-MiniLM-L6-v2).

Optimizations:
    - Model loaded once via st.cache_resource (survives reruns)
    - All section texts batched into a single model.encode() call
    - encode() uses show_progress_bar=False to avoid stdout noise
"""

from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    """
    Plain model loader — caching is applied in app.py via
    @st.cache_resource so the model loads only once per session.
    """
    return SentenceTransformer(_MODEL_NAME)


def compute_similarity(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two text passages.
    Returns a float in [0, 1].
    Accepts model as argument so the caller controls caching.
    """
    embeddings = model.encode(
        [text_a, text_b],
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=2,
    )
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(float(score), 4)


def compute_section_similarities(model: SentenceTransformer, resume_sections: dict[str, str], jd_sections: dict[str, str]) -> dict[str, float]:
    """
    Compute similarity scores for all sections in ONE batched encode call
    instead of 3-4 separate calls.
    """
    SECTIONS = ("skills", "experience", "education")

    pairs: list[tuple[str, str, str]] = []
    for section in SECTIONS:
        res_text = resume_sections.get(section, "").strip()
        jd_text  = jd_sections.get(section, "").strip()
        if res_text and jd_text:
            pairs.append((section, res_text, jd_text))

    results: dict[str, float | None] = {s: None for s in SECTIONS}
    if not pairs:
        return results

    all_texts = [p[1] for p in pairs] + [p[2] for p in pairs]
    embeddings = model.encode(
        all_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=len(all_texts),
    )

    n = len(pairs)
    for i, (section, _, _) in enumerate(pairs):
        score = util.cos_sim(embeddings[i], embeddings[n + i]).item()
        results[section] = round(float(score), 4)

    return results