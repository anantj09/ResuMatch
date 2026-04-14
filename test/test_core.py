"""
tests/test_core.py

Unit tests for parser, keywords, and similarity modules.
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.parser import parse_sections, _is_header_line, _classify_header
from src.keywords import normalise, keyword_gap_analysis
from src.similarity import compute_similarity


# Parser tests
class TestParser:
    def test_header_detection_skills(self):
        assert _is_header_line("SKILLS") is True
        assert _is_header_line("Technical Skills") is True

    def test_header_detection_experience(self):
        assert _is_header_line("Work Experience") is True
        assert _is_header_line("Professional Experience") is True

    def test_header_detection_education(self):
        assert _is_header_line("Education") is True
        assert _is_header_line("Academic Background") is True

    def test_non_header_long_line(self):
        long = "This is a very long line that definitely should not be treated as a section header in any case."
        assert _is_header_line(long) is False

    def test_classify_header(self):
        assert _classify_header("Technical Skills") == "skills"
        assert _classify_header("Work Experience") == "experience"
        assert _classify_header("Education") == "education"
        assert _classify_header("Random text") is None

    def test_parse_sections_returns_dict(self):
        text = """
        SKILLS
        Python, Machine Learning, NLP

        EXPERIENCE
        Software Engineer at ABC Corp

        EDUCATION
        B.Tech Computer Science
        """
        result = parse_sections(text)
        assert isinstance(result, dict)
        assert "skills" in result
        assert "experience" in result
        assert "education" in result

    def test_parse_sections_captures_content(self):
        text = "SKILLS\nPython\nJava\n\nEXPERIENCE\nEngineer at XYZ"
        result = parse_sections(text)
        assert "Python" in result["skills"]
        assert "Engineer" in result["experience"]


#Keyword tests
class TestKeywords:
    def test_normalise(self):
        assert normalise("  Machine Learning  ") == "machine learning"
        assert normalise("PYTHON") == "python"

    def test_gap_analysis_structure(self):
        resume = "Python machine learning data science scikit-learn pandas numpy deep learning"
        jd = "Python NLP transformers BERT scikit-learn TensorFlow data science"
        result = keyword_gap_analysis(resume, jd)
        assert "matched" in result
        assert "missing" in result
        assert "extra" in result
        assert "match_rate" in result
        assert 0.0 <= result["match_rate"] <= 1.0

    def test_gap_perfect_match(self):
        text = "Python machine learning data science"
        result = keyword_gap_analysis(text, text)
        assert result["match_rate"] == 1.0
        assert len(result["missing"]) == 0

    def test_gap_no_overlap(self):
        resume = "cooking baking gardening pottery weaving"
        jd = "Python NLP BERT transformers embeddings"
        result = keyword_gap_analysis(resume, jd)
        assert result["match_rate"] < 0.3


#Similarity tests
class TestSimilarity:
    def test_identical_texts(self):
        text = "Natural language processing and machine learning"
        score = compute_similarity(text, text)
        assert score >= 0.99

    def test_similar_texts(self):
        a = "Python developer with experience in machine learning"
        b = "Software engineer skilled in Python and ML"
        score = compute_similarity(a, b)
        assert score > 0.4

    def test_dissimilar_texts(self):
        a = "Experienced chef specializing in Italian cuisine and pastry arts"
        b = "NLP engineer building transformer models and text classifiers"
        score = compute_similarity(a, b)
        assert score < 0.6

    def test_score_in_range(self):
        score = compute_similarity("hello world", "goodbye moon")
        assert 0.0 <= score <= 1.0
