"""Core CID column classification engine."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .models import ClassificationResult, ClassifierConfig, ColumnInput
from .normalize import normalize_text
from .registry import load_cid_registry


def _exact_match(name: str, terms: List[str]) -> Tuple[bool, Optional[str]]:
    for term in terms:
        if name == term:
            return True, term
    return False, None


def _fuzzy_match(name: str, terms: List[str]) -> Tuple[float, Optional[str]]:
    best_score, best_term = 0.0, None
    for term in terms:
        score = max(
            fuzz.ratio(name, term),
            fuzz.partial_ratio(name, term),
            fuzz.token_sort_ratio(name, term),
        )
        if score > best_score:
            best_score, best_term = float(score), term
    return best_score, best_term


def _has_13_digit_pattern(samples: List[str], min_ratio: float = 0.7) -> bool:
    cleaned = [str(v).strip() for v in samples if str(v).strip()]
    if not cleaned:
        return False
    matched = sum(1 for v in cleaned if len(re.sub(r"\D", "", v)) == 13)
    return (matched / len(cleaned)) >= min_ratio


def _decide(result: ClassificationResult, config: ClassifierConfig) -> ClassificationResult:
    if result.lexical_exact:
        result.decision, result.reason, result.confidence = "auto_hash", "lexical_exact_match", 1.0
        return result

    if result.lexical_fuzzy_score >= config.fuzzy_auto_threshold:
        result.decision = "auto_hash"
        result.reason = "lexical_fuzzy_match"
        result.confidence = result.lexical_fuzzy_score / 100.0
        return result

    if result.lexical_fuzzy_score >= config.fuzzy_review_threshold:
        result.decision = "human_review"
        result.reason = "lexical_fuzzy_review"
        result.confidence = result.lexical_fuzzy_score / 100.0
        return result

    if result.semantic_score is not None:
        if result.semantic_score >= config.semantic_auto_threshold:
            if result.generic_name_risk and not result.value_pattern_signal:
                result.decision = "human_review"
                result.reason = "semantic_high_but_generic_name"
            else:
                result.decision = "auto_hash"
                result.reason = "semantic_high_confidence"
            result.confidence = result.semantic_score
            return result

        if result.semantic_score >= config.semantic_review_threshold:
            result.decision = "human_review"
            result.reason = "semantic_mid_confidence"
            result.confidence = result.semantic_score
            return result

    result.decision = "pass"
    result.reason = "not_cid"
    result.confidence = result.semantic_score or 0.0
    return result


class ColumnClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        if load_dotenv is not None:
            load_dotenv()

        self.config = config or ClassifierConfig()
        self.registry = load_cid_registry(self.config.registry_path)

        self._cid_terms = [self._normalize(x) for x in self.registry.cid_terms]
        self._generic_terms = [self._normalize(x) for x in self.registry.generic_terms]
        self._semantic_refs = [self._normalize(x) for x in self.registry.cid_semantic_references]

        self._model = None
        self._hf_client = None
        self._ref_embeddings = None
        self._semantic_backend = "disabled"

        self._init_semantic_backend()

    def _normalize(self, text: str) -> str:
        return normalize_text(text, self.registry.replacements)

    def _init_semantic_backend(self) -> None:
        backend = self.config.semantic_backend
        if backend not in {"auto", "local", "hf_api", "disabled"}:
            raise ValueError(f"Unsupported semantic backend: {backend}")

        if backend == "disabled":
            return

        if backend in {"auto", "local"} and SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.config.embedding_model_name)
                self._ref_embeddings = self._model.encode(
                    self._semantic_refs,
                    normalize_embeddings=True,
                )
                self._semantic_backend = "local"
                return
            except Exception:
                if backend == "local":
                    raise

        token = self.config.hf_api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if backend in {"auto", "hf_api"} and InferenceClient is not None and token:
            try:
                self._hf_client = InferenceClient(
                    provider=self.config.hf_api_provider,
                    api_key=token,
                )
                self._ref_embeddings = self._embed_with_hf_api(self._semantic_refs)
                self._semantic_backend = "hf_api"
                return
            except Exception:
                if backend == "hf_api":
                    raise

        if backend in {"local", "hf_api"}:
            raise RuntimeError(
                f"Unable to initialize semantic backend '{backend}'. "
                "Check installed packages, credentials, and network access."
            )

    def _embed_with_hf_api(self, texts: List[str]) -> np.ndarray:
        if self._hf_client is None:
            raise RuntimeError("Hugging Face API client is not initialized.")

        vectors = self._hf_client.feature_extraction(
            texts,
            model=self.config.embedding_model_name,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def _semantic_score(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        if self._ref_embeddings is None:
            return None, None

        if self._semantic_backend == "local":
            emb = self._model.encode([text], normalize_embeddings=True)[0]
        elif self._semantic_backend == "hf_api":
            emb = self._embed_with_hf_api([text])[0]
        else:
            return None, None

        scores = np.dot(self._ref_embeddings, emb)
        idx = int(np.argmax(scores))
        return float(scores[idx]), self._semantic_refs[idx]

    def classify(self, column: ColumnInput) -> ClassificationResult:
        name = self._normalize(column.column_name)
        result = ClassificationResult(column_name=column.column_name, normalized_name=name)

        exact, exact_term = _exact_match(name, self._cid_terms)
        result.lexical_exact, result.lexical_exact_term = exact, exact_term
        if exact:
            result.metadata["matched_stage"] = "lexical_exact"
            return _decide(result, self.config)

        fuzzy_score, fuzzy_term = _fuzzy_match(name, self._cid_terms)
        result.lexical_fuzzy_score, result.lexical_fuzzy_term = fuzzy_score, fuzzy_term
        if fuzzy_score >= self.config.fuzzy_review_threshold:
            result.metadata["matched_stage"] = "lexical_fuzzy"
            return _decide(result, self.config)

        sem_score, sem_term = self._semantic_score(name)
        result.semantic_score, result.semantic_term = sem_score, sem_term

        result.generic_name_risk = name in self._generic_terms
        result.value_pattern_signal = _has_13_digit_pattern(column.sample_values)

        result.metadata.update({
            "matched_stage": "semantic",
            "sample_values_count": len(column.sample_values),
        })

        return _decide(result, self.config)
