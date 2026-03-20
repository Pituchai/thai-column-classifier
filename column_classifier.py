"""Backward-compatible public entrypoint.

This module re-exports the classifier API after refactoring into the
`classifier/` package.
"""

from classifier import (
    CIDRegistry,
    ClassifierConfig,
    ClassificationResult,
    ColumnClassifier,
    ColumnInput,
    load_cid_registry,
)

__all__ = [
    "ColumnClassifier",
    "ColumnInput",
    "ClassificationResult",
    "ClassifierConfig",
    "CIDRegistry",
    "load_cid_registry",
]
