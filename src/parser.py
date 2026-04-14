"""
src/parser.py

Rule-based section segmentation for resumes and job descriptions.
Identifies Skills, Experience, and Education sections using
header keyword matching + heuristic boundary detection.
"""

import re
from typing import Optional

# Canonical section names - regex patterns for their headers
SECTION_PATTERNS: dict[str, list[str]] = {
    "skills": [
        r"\bskills?\b",
        r"\btechnical\s+skills?\b",
        r"\bcore\s+competenc",
        r"\btechnologies\b",
        r"\btools?\b",
        r"\bproficienc",
    ],
    "experience": [
        r"\bwork\s+experience\b",
        r"\bprofessional\s+experience\b",
        r"\bemployment\b",
        r"\bexperience\b",
        r"\bwork\s+history\b",
        r"\binternship",
    ],
    "education": [
        r"\beducation\b",
        r"\bacademic\b",
        r"\bqualification",
        r"\bdegree\b",
        r"\buniversity\b",
        r"\bcollege\b",
    ],
}

# Section headers that signal a new section (for boundary detection)
_ALL_HEADERS = [p for patterns in SECTION_PATTERNS.values() for p in patterns]


def _is_header_line(line: str) -> bool:
    """Return True if a line looks like a section header."""
    cleaned = line.strip()
    # Headers are typically short, possibly ALL-CAPS or Title Case
    if len(cleaned) > 60:
        return False
    for pat in _ALL_HEADERS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return True
    return False


def _classify_header(line: str) -> Optional[str]:
    """Return the canonical section name for a header line, or None."""
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                return section
    return None


def parse_sections(text: str) -> dict[str, str]:
    """
    Split raw text into labelled sections.
    Returns dict with keys: 'skills', 'experience', 'education', 'other'.
    Missing sections get an empty string.
    """
    lines = text.splitlines()
    sections: dict[str, list[str]] = {
        "skills": [],
        "experience": [],
        "education": [],
        "other": [],
    }
    current_section = "other"

    for line in lines:
        if _is_header_line(line):
            label = _classify_header(line)
            if label:
                current_section = label
                continue  # skip the header line itself
        sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_full_text_sections(resume_text: str, jd_text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse both resume and JD into sections."""
    return parse_sections(resume_text), parse_sections(jd_text)
