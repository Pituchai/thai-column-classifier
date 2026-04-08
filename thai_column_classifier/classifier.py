"""
classifier.py
─────────────
Unified entry point that wraps IDColumnClassifier and SensitiveColumnClassifier.

Usage:
    from thai_column_classifier import ThaiColumnClassifier, OllamaProvider

    clf = ThaiColumnClassifier(llm_provider=OllamaProvider("llama3.2"))

    # Single column
    result = clf.classify("cid", samples=["1234567890123", ...])
    print(result.decision)   # auto_hash | masking | partial_masking | pass
    print(result.type)       # CID | FULLNAME | ADDRESS_FULL | ... | None

    # Whole DataFrame
    results = clf.classify_dataframe(df)
    for col, r in results.items():
        print(col, r.decision)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


from .thai_id_column_detector import (
    IDColumnClassifier,
    ColumnInput as _IDInput,
    ClassifierConfig,
)
from .thai_sensitive_column_detector import (
    SensitiveColumnClassifier,
    ColumnInput as _SensitiveInput,
)


@dataclass
class ColumnResult:
    """Unified result returned by ThaiColumnClassifier."""

    column_name: str

    decision: str = "pass"
    """auto_hash | human_review | masking | partial_masking | pass"""

    type: Optional[str] = None
    """CID | FULLNAME | FIRSTNAME | LASTNAME | PREFIX | EMAIL | ADDRESS_SHORT | ADDRESS_FULL | GEO | None"""

    reason: str = ""
    confidence: float = 0.0
    detector: str = ""
    """'cid' | 'sensitive' | ''"""

    metadata: Dict[str, Any] = field(default_factory=dict)


class ThaiColumnClassifier:
    """
    Unified classifier that runs CID detection first, then sensitive-data
    detection on columns that pass the CID check.

    Args:
        llm_provider:       OpenAIProvider | OllamaProvider | ClaudeProvider | HFLLMProvider | None
        semantic_provider:  LocalSemanticProvider | HFSemanticProvider | None
        cid_config:         ClassifierConfig for fine-tuning ID detection thresholds
        fuzzy_threshold:    Fuzzy match threshold for sensitive detection (default 92.0)
        semantic_threshold: Semantic similarity threshold for sensitive detection (default 0.85)
        llm_threshold:      LLM confidence threshold for sensitive detection (default 0.7)
    """

    def __init__(
        self,
        llm_provider=None,
        semantic_provider=None,
        cid_config: Optional[ClassifierConfig] = None,
        fuzzy_threshold: float = 92.0,
        semantic_threshold: float = 0.85,
        llm_threshold: float = 0.7,
    ):
        self._cid = IDColumnClassifier(config=cid_config or ClassifierConfig())
        self._sensitive = SensitiveColumnClassifier(
            fuzzy_threshold=fuzzy_threshold,
            semantic_threshold=semantic_threshold,
            llm_threshold=llm_threshold,
            semantic_provider=semantic_provider,
            llm_provider=llm_provider,
        )

    def classify(self, column_name: str, samples: Optional[List[str]] = None) -> ColumnResult:
        """
        Classify a single column.

        Args:
            column_name: Name of the column.
            samples:     Optional list of sample values from the column.

        Returns:
            ColumnResult with decision, type, reason, confidence, detector.
        """
        samples = samples or []

        # Stage 1 — CID
        cid_result = self._cid.classify(_IDInput(column_name=column_name, sample_values=samples))
        if cid_result.decision != "pass":
            return ColumnResult(
                column_name=column_name,
                decision=cid_result.decision,
                type="CID",
                reason=cid_result.reason,
                confidence=round(cid_result.confidence, 3),
                detector="cid",
            )

        # Stage 2 — Sensitive
        sen_result = self._sensitive.classify(_SensitiveInput(column_name=column_name, sample_values=samples))
        return ColumnResult(
            column_name=column_name,
            decision=sen_result.decision,
            type=sen_result.sensitive_type,
            reason=sen_result.reason,
            confidence=round(sen_result.confidence, 3),
            detector="sensitive",
            metadata=sen_result.metadata,
        )

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: int = 20,
        show_progress: bool = False,
    ) -> Dict[str, ColumnResult]:
        """
        Classify all columns in a DataFrame.

        Args:
            df:            Input DataFrame.
            sample_size:   Number of non-null sample values to pull per column (default 20).
            show_progress: Show a tqdm progress bar (requires tqdm installed).

        Returns:
            Dict mapping column_name -> ColumnResult.
        """
        if show_progress:
            if _tqdm is None:
                raise ImportError("pip install tqdm")
            columns = _tqdm(df.columns.tolist(), desc="Classifying columns", unit="col")
        else:
            columns = df.columns

        results: Dict[str, ColumnResult] = {}
        for col in columns:
            samples = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .pipe(lambda s: s[s != ""])
                .head(sample_size)
                .tolist()
            )
            results[col] = self.classify(col, samples)
        return results
