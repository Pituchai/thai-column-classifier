"""
column_classifier.py
────────────────────
Single-file library for classifying sensitive columns (CID / เลขบัตรประชาชน).

Usage:
    from column_classifier import IDColumnClassifier, ColumnInput, ClassifierConfig

    clf = IDColumnClassifier()
    result = clf.classify(ColumnInput(column_name="เลขบัตรประชาชน", sample_values=["1101700203451"]))
    print(result.decision)   # auto_hash | human_review | pass
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

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


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

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

    value_pattern_signal: bool = False

    decision: str = "pass"          # auto_hash | human_review | pass
    reason: str = "not_classified"
    confidence: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    fuzzy_auto_threshold: float = 92.0
    semantic_auto_threshold: float = 0.93

    semantic_backend: str = "local"  # auto | local | hf_api | disabled
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hf_api_provider: str = "hf-inference"
    hf_api_token: Optional[str] = None


# ─────────────────────────────────────────────
# Patterns & Normalizer
# ─────────────────────────────────────────────



# CID_TERMS เป็นรายการคำที่ใช้บอกว่า “ชื่อนี้น่าจะเป็นคอลัมน์เลขบัตรประชาชน” ใช้สำหรับ exact match และ fuzzy match
_CID_TERMS = [
    "เลขบัตรประชาชน", "เลขบัตร", "เลขประจำตัวประชาชน", "เลขปชช",
    "รหัสบัตรประชาชน", "รหัสปชช", "รหัสบัตรปชช",
    "รหัสประจำตัวประชาชน", "รหัสประจำตัว",
    "บัตรประชาชน", "หมายเลขบัตรประชาชน", "เลขประจำตัว", "ประชาชน_id",
    "เลขที่บัตรประจำตัวประชาชน", "เลขที่บัตรประชาชน",
    "cid", "citizen id", "citizen_id", "citizenid",
    "citizen identification number", "national id", "national_id",
    "national identification number", "id card", "id_card",
    "personal id", "personal identification number",
]

# CID_SEMANTIC_REFERENCES เป็นรายการคำอ้างอิงสำหรับ semantic search ใช้ดูว่า “ความหมาย” ของชื่อคอลัมน์คล้ายกับคำที่เกี่ยวกับเลขบัตรประชาชนไหม ไม่ได้ดูแค่ตัวสะกดใกล้กัน แต่ดูความหมายโดยรวม
_CID_SEMANTIC_REFERENCES = [
    "เลขบัตรประชาชน", "เลขประจำตัวประชาชน", "หมายเลขบัตรประชาชน",
    "citizen identification number", "national identification number",
    "personal identification number", "thai national id",
]

# REPLACEMENTS เป็นตัวแปลงคำก่อนเอาไป match ใช้แก้คำสะกดติดกัน คำย่อ หรือคำพิมพ์ผิดที่พบบ่อย ใช้สำหรับ normalize ชื่อคอลัมน์ก่อนเอาไปเทียบกับ CID_TERMS และ CID_SEMANTIC_REFERENCES เพื่อเพิ่มโอกาสในการจับคู่ เช่น "citizenid" → "citizen id", "เลขบตรประชาชน" → "เลขบัตรประชาชน"
_REPLACEMENTS = {
    "citizenid": "citizen id",
    "nationalid": "national id",
    "idcard": "id card",
    "เลขบตรประชาชน": "เลขบัตรประชาชน",
    "เลขบัตรปชช": "เลขบัตรประชาชน",
    "เลขบัตร ประชาชน" : "เลขบัตรประชาชน",
}

# lowercase, remove special chars, normalize whitespace, and apply replacements
def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[_\-/]", " ", text)
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _REPLACEMENTS.get(text, text)


# ─────────────────────────────────────────────
# Engine (Lexical + Semantic + Rules + Decision)
# ─────────────────────────────────────────────
# name คือค่าที่เราคิดว่าเรากำลังจะทำ auto_hash หรือ human_review หรือ pass กับคอลัมน์นี้อยู่
# term คือค่าที่เรามีอยู่ในมือแล้ว และอยากจะเทียบกับ name ว่ามันใกล้เคียงกันแค่ไหน

# เทียบ 1:1
def _exact_match(name: str, terms: List[str]) -> Tuple[bool, Optional[str]]:
    for term in terms:
        if name == term:
            return True, term
    return False, None

# เทียบแบบ fuzzy match ใช้หลายวิธี แล้วเอาคะแนนสูงสุดมาใช้
def _fuzzy_match(name: str, terms: List[str]) -> Tuple[float, Optional[str]]:
    best_score, best_term = 0.0, None
    for term in terms:
        scores = [
            # เทียบตรง ๆ เหมาะกับ string ยาวเท่ากัน
            fuzz.ratio(name, term),
            # เช็ค token order เหมาะกับ "คำสลับตำแหน่ง"
            fuzz.token_sort_ratio(name, term),
        ]
        # ใช้ partial_ratio เฉพาะตอนที่ name ยาวกว่า term เท่านั้น
        # เพราะ partial_ratio จะเอา string ที่สั้นกว่าไปวิ่งหา match ในทุกตำแหน่งของ string ที่ยาวกว่า
        # ถ้าไม่เช็ค จะเกิด false positive เช่น "id" match "cid" หรือ "number" match "identification_number"
        # การเช็คนี้การันตีว่า term เป็น "subset" ที่ถูกค้นหาใน name เสมอ ไม่ใช่กลับกัน
        if len(name) > len(term):
            scores.append(fuzz.partial_ratio(name, term))
        score = max(scores)
        if score > best_score:
            best_score, best_term = float(score), term
    return best_score, best_term


_NULL_STR = {"", "none", "nan", "null", "na", "n/a", "nat", "<na>"}


def _thai_cid_checksum(value: str) -> bool:
    """ตรวจสอบ checksum ของเลขบัตรประชาชนไทย 13 หลัก"""
    total = sum(int(value[i]) * (13 - i) for i in range(12))
    return (11 - (total % 11)) % 10 == int(value[12])


# ตรวจสอบ pattern ใน sample values — นับเฉพาะค่าที่ไม่ใช่ null, ตรงกับ pattern ตัวเลข 13 หลักพอดี และผ่าน checksum
def _has_13_digit_pattern(samples: List[str], min_ratio: float = 0.25) -> bool:
    non_null = [str(v).strip() for v in samples if str(v).strip().lower() not in _NULL_STR]
    if not non_null:
        return False
    matched = sum(
        1 for v in non_null
        if re.fullmatch(r"\d{13}", re.sub(r"[\s\-]", "", v))
        and _thai_cid_checksum(re.sub(r"[\s\-]", "", v))
    )
    return (matched / len(non_null)) >= min_ratio


# ตัดสินใจขั้นสุดท้าย: exact/fuzzy → auto_hash หรือ human_review, semantic → auto_hash หรือ human_review, 13-digit guardrail → auto_hash
def _decide(result: ClassificationResult, config: ClassifierConfig) -> ClassificationResult:
    # exact match
    if result.lexical_exact:
        result.decision, result.reason, result.confidence = "auto_hash", "lexical_exact_match", 1.0
        return result

    # fuzzy match
    if result.lexical_fuzzy_score >= config.fuzzy_auto_threshold:
        result.decision = "auto_hash"
        result.reason = "lexical_fuzzy_match"
        result.confidence = result.lexical_fuzzy_score / 100.0
        return result

    # semantic search
    if result.semantic_score is not None:
        if result.semantic_score >= config.semantic_auto_threshold:
            result.decision = "auto_hash"
            result.reason = "semantic_high_confidence"
            result.confidence = result.semantic_score
            return result

    # value pattern
    if result.value_pattern_signal:
        result.decision = "auto_hash"
        result.reason = "value_pattern_13digit"
        result.confidence = 1.0
        return result

    result.decision = "pass"
    result.reason = "not_cid"
    result.confidence = result.semantic_score or 0.0
    return result


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class IDColumnClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        if load_dotenv is not None:
            load_dotenv()

        self.config = config or ClassifierConfig()

        self._cid_terms = [_normalize(x) for x in _CID_TERMS]
        self._semantic_refs = [_normalize(x) for x in _CID_SEMANTIC_REFERENCES]

        self._model = None
        self._hf_client = None
        self._ref_embeddings = None
        self._semantic_backend = "disabled"

        self._init_semantic_backend()

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
        name = _normalize(column.column_name)
        result = ClassificationResult(column_name=column.column_name, normalized_name=name)

        # Stage 1 — Exact
        exact, exact_term = _exact_match(name, self._cid_terms)
        result.lexical_exact, result.lexical_exact_term = exact, exact_term
        if exact:
            result.metadata["matched_stage"] = "lexical_exact"
            return _decide(result, self.config)

        # Stage 2 — Fuzzy
        fuzzy_score, fuzzy_term = _fuzzy_match(name, self._cid_terms)
        result.lexical_fuzzy_score, result.lexical_fuzzy_term = fuzzy_score, fuzzy_term
        # Stage 3 — Semantic
        sem_score, sem_term = self._semantic_score(name)
        result.semantic_score, result.semantic_term = sem_score, sem_term

        # Stage 4 — Guardrail
        result.value_pattern_signal = _has_13_digit_pattern(column.sample_values)

        result.metadata.update({
            "matched_stage": "semantic",
            "sample_values_count": len(column.sample_values),
        })

        return _decide(result, self.config)
