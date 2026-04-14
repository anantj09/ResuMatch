from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def _extract_tfidf(text: str, top_n: int = 20) -> list[str]:
    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=200,
        )
        vec.fit([text])
        scores = vec.transform([text]).toarray()[0]
        top_idx = scores.argsort()[::-1][:top_n]
        return [vec.get_feature_names_out()[i] for i in top_idx]
    except Exception:
        return []

def extract_keywords(text: str, top_n: int = 20, model=None) -> list[str]:
    return _extract_tfidf(text, top_n)

def normalise(kw: str) -> str:
    return re.sub(r"\s+", " ", kw.lower().strip())

def keyword_gap_analysis(resume_text: str, jd_text: str, top_n: int = 25, model=None) -> dict:
    jd_kws  = [normalise(k) for k in _extract_tfidf(jd_text, top_n)]
    res_kws = [normalise(k) for k in _extract_tfidf(resume_text, top_n)]
    jd_set, res_set = set(jd_kws), set(res_kws)

    def soft_match(a: str, b_set: set[str]) -> bool:
        if a in b_set:
            return True
        return any(a in b or b in a for b in b_set)

    matched = [k for k in jd_kws if soft_match(k, res_set)]
    missing = [k for k in jd_kws if not soft_match(k, res_set)]
    extra   = [k for k in res_kws if not soft_match(k, jd_set)]

    return {
        "matched": matched, "missing": missing, "extra": extra,
        "jd_keywords": jd_kws, "resume_keywords": res_kws,
        "match_rate": round(len(matched) / max(len(jd_kws), 1), 4),
    }