"""Text normalization helpers for column classification."""

import re
from typing import Dict


def normalize_text(text: str, replacements: Dict[str, str]) -> str:
    if not text:
        return ""
    normalized = text.strip().lower()
    normalized = re.sub(r"[_\-/]", " ", normalized)
    normalized = re.sub(r"[^\w\sก-๙]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return replacements.get(normalized, normalized)
