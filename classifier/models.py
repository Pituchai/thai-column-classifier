"""Dataclasses used by the CID classifier."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnInput:
    column_name: str
    sample_values: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    column_name: str
    normalized_name: str

    lexical_exact: bool = False
    lexical_exact_term: Optional[str] = None

    lexical_fuzzy_score: float = 0.0
    lexical_fuzzy_term: Optional[str] = None

    semantic_score: Optional[float] = None
    semantic_term: Optional[str] = None

    generic_name_risk: bool = False
    value_pattern_signal: bool = False

    decision: str = "pass"  # auto_hash | human_review | pass
    reason: str = "not_classified"
    confidence: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    fuzzy_auto_threshold: float = 92.0
    fuzzy_review_threshold: float = 85.0

    semantic_auto_threshold: float = 0.93
    semantic_review_threshold: float = 0.75

    use_value_pattern_guardrail: bool = True

    semantic_backend: str = "auto"  # auto | local | hf_api | disabled
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hf_api_provider: str = "hf-inference"
    hf_api_token: Optional[str] = None

    # Registry path is optional; defaults to bundled YAML.
    registry_path: Optional[str] = None
