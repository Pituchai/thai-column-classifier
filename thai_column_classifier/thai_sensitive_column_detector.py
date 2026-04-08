"""
thai_sensitive_column_detector.py
──────────────────────────────────
Detects sensitive columns for Thai personal data protection (PDPA).

Sensitive types:
    FULLNAME, PREFIX, FIRSTNAME, LASTNAME  → masking
    EMAIL                                  → masking
    ADDRESS_SHORT (house no, soi, road)    → masking
    ADDRESS_FULL  (full address column)    → partial_masking
    GEO           (lat/lon, ละติจูด/ลองจิจูด) → masking

Output decisions:
    masking         — mask entire value (****)
    partial_masking — mask street-level part, keep district/province
    pass            — not sensitive

Detection pipeline (per column):
    Stage 1: Exact Match
    Stage 2: Fuzzy Match  (ratio, partial_ratio, token_sort_ratio)
    Stage 3: Semantic     (pluggable: LocalSemanticProvider | HFSemanticProvider)
    Stage 4: LLM          (pluggable: OpenAIProvider | OllamaProvider | ClaudeProvider | HFLLMProvider)

Usage:
    from thai_sensitive_column_detector import SensitiveColumnClassifier, ColumnInput
    from thai_sensitive_column_detector import OpenAIProvider, LocalSemanticProvider

    clf = SensitiveColumnClassifier(
        llm_provider=OpenAIProvider(api_key="sk-..."),
        semantic_provider=LocalSemanticProvider(),
    )
    result = clf.classify(ColumnInput(column_name="ชื่อ-นามสกุล", sample_values=["สมชาย ใจดี"]))
    print(result.decision)  # masking
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from rapidfuzz import fuzz

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from huggingface_hub import InferenceClient as HFInferenceClient
except ImportError:
    HFInferenceClient = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import ollama as ollama_client
except ImportError:
    ollama_client = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnInput:
    column_name: str
    sample_values: List[str] = field(default_factory=list)


@dataclass
class SensitiveClassificationResult:
    column_name: str
    normalized_name: str

    sensitive_type: Optional[str] = None       # FULLNAME | PREFIX | FIRSTNAME | LASTNAME | EMAIL | ADDRESS_SHORT | ADDRESS_FULL | GEO

    lexical_exact: bool = False
    lexical_exact_term: Optional[str] = None

    lexical_fuzzy_score: float = 0.0
    lexical_fuzzy_term: Optional[str] = None

    semantic_score: Optional[float] = None
    semantic_category: Optional[str] = None

    value_pattern_signal: bool = False

    decision: str = "pass"                     # masking | partial_masking | pass
    reason: str = "not_classified"
    confidence: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Term Lists
# ─────────────────────────────────────────────────────────────────────────────

_SENSITIVE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "FULLNAME": {
        "terms": [
            "ชื่อ-นามสกุล", "ชื่อนามสกุล", "ชื่อ-สกุล", "ชื่อสกุล",
            "ชื่อ-สกุล ผู้สูงอายุ", "ชื่อ-สกุล ผู้เสียชีวิต",
            "fullname", "full name", "full_name",
            "person_name", "personname", "rgt_name",
            "namethai", "name_thai", "sufferername",
            "first name last name", "firstname lastname",
        ],
        "decision": "masking",
    },
    "PREFIX": {
        "terms": [
            "คำนำหน้า", "คำนำหน้านาม", "คำนำหน้าชื่อ",
            "คำนำหน้าผู้ขอรับความช่วยเหลือ",
            "คำนำหน้าชื่อ กรรมการ หุ้นส่วน",
            "คำนำหน้าชื่อ หุ้นส่วนผู้จัดการ",
            "prefix", "prefix_name", "title", "titles",
            "title_id", "title_desc", "name title",
            "personnametitletextth",
        ],
        "decision": "pass",
    },
    "FIRSTNAME": {
        "terms": [
            "ชื่อ", "ชื่อต้น", "ชื่อจริง",
            "ชื่อ ผู้กู้", "ชื่อ กรรมการ หุ้นส่วน",
            "ชื่อผู้ขอรับความช่วยเหลือ", "ชื่อ หุ้นส่วนผู้จัดการ",
            "fname", "first_name", "firstname", "given_name", "givenname",
            "encrypted_name", "encrypted first name",
            "personfirstnameth", "personmiddlenameth",
        ],
        "decision": "masking",
    },
    "LASTNAME": {
        "terms": [
            "นามสกุล", "สกุล",
            "นามสกุล ผู้กู้", "นามสกุล กรรมการ หุ้นส่วน",
            "นามสกุลผู้ขอรับความช่วยเหลือ", "นามสกุล หุ้นส่วนผู้จัดการ",
            "lname", "last_name", "lastname", "surname", "family_name", "familyname",
            "encrypted_lastname", "encrypted last name",
            "personlastnameth", "personnamesuffixth",
        ],
        "decision": "masking",
    },
    "EMAIL": {
        "terms": [
            "อีเมล", "อีเมล์",
            "email", "e_mail", "e mail", "mail",
            "email_address", "emailaddress",
        ],
        "decision": "masking",
    },
    "ADDRESS_SHORT": {
        "terms": [
            "บ้านเลขที่", "เลขที่บ้าน", "เลขที่", "หมู่ที่", "หมู่บ้าน", "หมู่บ้าน ชุมชน", "ชุมชน", "ซอย", "ถนน",
            "house no", "house number", "houseno", "housenumber",
            "address no", "address number",
            "street", "street no", "street number", "street address",
            "road", "moo", "soi", "alley",
        ],
        "decision": "masking",
    },
    "ADDRESS_FULL": {
        "terms": [
            "ที่อยู่", "ที่อยู่เต็ม", "ที่อยู่ปัจจุบัน", "ที่อยู่ตามทะเบียนบ้าน",
            "ที่อยู่ผู้ขอรับความช่วยเหลือ", "ที่อยู่ผู้กู้", "ที่อยู่ผู้สมัคร",
            "address", "full address", "home address", "mailing address",
            "current address", "residential address", "contact address",
            "addr", "address1", "address2",
        ],
        "decision": "partial_masking",
    },
    "GEO": {
        "terms": [
            "ละติจูด", "ลองจิจูด", "พิกัด",
            "lat", "lon", "latitude", "longitude",
            "lat lon", "latitude longitude",
        ],
        "decision": "masking",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────────

# name คือค่าที่เราคิดว่าเรากำลังจะทำ masking หรือ pass กับคอลัมน์นี้อยู่
# term คือค่าที่เรามีอยู่ในมือแล้ว และอยากจะเทียบกับ name ว่ามันใกล้เคียงกันแค่ไหน

_REPLACEMENTS = {
    "firstname": "first name",
    "lastname": "last name",
    "fullname": "full name",
    "emailaddress": "email address",
    "givenname": "given name",
    "givename": "given name",   # GiveName → givename (single n)
    "familyname": "family name",
    "houseno": "house no",
    "housenumber": "house number",
    "streetno": "street no",
    "streetaddress": "street address",
    "homeaddress": "home address",
    "mailingaddress": "mailing address",
    "currentaddress": "current address",
    "residentialaddress": "residential address",
    "contactaddress": "contact address",
    "fulladdress": "full address",
}


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[_\-/]", " ", text)
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _REPLACEMENTS.get(text, text)


# ─────────────────────────────────────────────────────────────────────────────
# Matchers
# ─────────────────────────────────────────────────────────────────────────────

def _exact_match(name: str, terms: List[str]) -> Tuple[bool, Optional[str]]:
    for term in terms:
        if name == term:
            return True, term
    return False, None


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
        # เพราะ partial_ratio จะเอา term ไปวิ่งหา match ในทุกตำแหน่งของ name
        # การันตีว่า term เป็น "subset" ที่ถูกค้นหาใน name เสมอ ไม่ใช่กลับกัน
        # เพิ่มเงื่อนไข coverage >= 0.6 เพื่อป้องกัน false positive
        # เช่น "ที่อยู่" (7 chars) ใน "ลักษณะที่อยู่อาศัย" (19 chars) = 0.37 → ไม่ใช้ partial_ratio
        coverage = len(term) / len(name)
        if len(name) > len(term) and coverage >= 0.6:
            scores.append(fuzz.partial_ratio(name, term))
        score = max(scores)
        if score > best_score:
            best_score, best_term = float(score), term
    return best_score, best_term


_NULL_STR = {"", "none", "nan", "null", "na", "n/a", "<na>"}
_EMAIL_PATTERN = re.compile(r"[\w.+\-]+@[\w.\-]+\.\w+")

# Thai address component keywords — ถ้า value มีคำเหล่านี้แสดงว่ามี component ของที่อยู่เต็ม
_THAI_ADDR_KEYWORDS = {"ถนน", "ตำบล", "แขวง", "อำเภอ", "เขต", "จังหวัด"}
# English road keywords (case-insensitive)
_EN_ROAD_PATTERN = re.compile(
    r"\b(rd|road|st|street|ave|avenue|blvd|boulevard|moo|soi|alley)\b",
    re.IGNORECASE,
)
# Thai/Thai-style postal code (5 digits, standalone)
_POSTAL_CODE_PATTERN = re.compile(r"\b\d{5}\b")


def _has_email_pattern(samples: List[str], min_ratio: float = 0.25) -> bool:
    non_null = [v.strip() for v in samples if v.strip().lower() not in _NULL_STR]
    if not non_null:
        return False
    matched = sum(1 for v in non_null if _EMAIL_PATTERN.fullmatch(v))
    return (matched / len(non_null)) >= min_ratio


def _has_full_address_pattern(samples: List[str], min_ratio: float = 0.25) -> bool:
    """ตรวจจาก sample values ว่าเป็น full address หรือไม่
    เงื่อนไข: value ต้องมี token >= 4 คำ (ยาว) และมี signal อย่างน้อย 1 ข้อ:
      - Thai: มี keyword ของ address component (ถนน/ตำบล/แขวง/อำเภอ/เขต/จังหวัด)
      - English: มี road keyword + postal code
    """
    non_null = [v.strip() for v in samples if v.strip().lower() not in _NULL_STR]
    if not non_null:
        return False

    def _is_full_address(v: str) -> bool:
        if len(v.split()) < 4:
            return False
        if any(kw in v for kw in _THAI_ADDR_KEYWORDS):
            return True
        if _EN_ROAD_PATTERN.search(v) and _POSTAL_CODE_PATTERN.search(v):
            return True
        return False

    matched = sum(1 for v in non_null if _is_full_address(v))
    return (matched / len(non_null)) >= min_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Providers (pluggable)
# ─────────────────────────────────────────────────────────────────────────────

_SEMANTIC_REFERENCES = {
    "FULLNAME":       "ชื่อและนามสกุล full name person name complete name ชื่อ-นามสกุล",
    "PREFIX":         "คำนำหน้าชื่อ name title prefix นาย นาง นางสาว",
    "FIRSTNAME":      "ชื่อจริง first name given name ชื่อต้น",
    "LASTNAME":       "นามสกุล last name family name surname",
    "EMAIL":          "อีเมล email address e-mail electronic mail",
    "ADDRESS_SHORT":  "บ้านเลขที่ ซอย ถนน หมู่ที่ house number street road soi moo alley",
    "ADDRESS_FULL":   "ที่อยู่ ที่อยู่เต็ม ที่อยู่ปัจจุบัน full address home address mailing address residential address",
    "GEO":            "ละติจูด ลองจิจูด พิกัด latitude longitude lat lon geographic coordinates",
}


class LocalSemanticProvider:
    """Semantic provider using local SentenceTransformer model (~500MB download on first use)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if SentenceTransformer is None:
            raise ImportError("pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._keys = list(_SEMANTIC_REFERENCES.keys())
        self._ref_embeddings = self._model.encode(
            list(_SEMANTIC_REFERENCES.values()),
            normalize_embeddings=True,
        )

    def score(self, name: str) -> Tuple[float, str]:
        emb = self._model.encode([name], normalize_embeddings=True)[0]
        scores = np.dot(self._ref_embeddings, emb)
        idx = int(np.argmax(scores))
        return float(scores[idx]), self._keys[idx]


class HFSemanticProvider:
    """Semantic provider using HuggingFace Inference API (no local model needed)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if HFInferenceClient is None:
            raise ImportError("pip install huggingface_hub")
        self._client = HFInferenceClient(api_key=api_key or os.getenv("HF_TOKEN"))
        self._model_name = model_name
        self._keys = list(_SEMANTIC_REFERENCES.keys())
        self._ref_embeddings = self._embed(list(_SEMANTIC_REFERENCES.values()))

    def _embed(self, texts: List[str]) -> np.ndarray:
        arr = np.asarray(
            self._client.feature_extraction(texts, model=self._model_name),
            dtype=np.float32,
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    def score(self, name: str) -> Tuple[float, str]:
        emb = self._embed([name])[0]
        scores = np.dot(self._ref_embeddings, emb)
        idx = int(np.argmax(scores))
        return float(scores[idx]), self._keys[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LLM Providers (pluggable)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are a data privacy expert specializing in Thai personal data classification (PDPA).

Given a column name and optional sample values from a dataset, determine if the column contains sensitive personal data.

Sensitive types:
- FULLNAME       : full name (ชื่อ-นามสกุล, full name, person name)
- PREFIX         : name prefix/title (คำนำหน้า, title, นาย/นาง/นางสาว)
- FIRSTNAME      : first name only (ชื่อ, fname, given name)
- LASTNAME       : last name only (นามสกุล, lname, surname)
- EMAIL          : email address
- ADDRESS_SHORT  : specific street-level field (บ้านเลขที่, ซอย, ถนน, house number, street, road, soi)
- ADDRESS_FULL   : full address column (ที่อยู่, address, home address, mailing address)
- GEO            : geographic coordinates (ละติจูด, ลองจิจูด, พิกัด, lat, lon, latitude, longitude)

Decisions:
- masking         : mask entire value (FULLNAME, FIRSTNAME, LASTNAME, EMAIL, ADDRESS_SHORT, GEO)
- partial_masking : mask street-level part only, keep district/province (ADDRESS_FULL), ตั้งแต่ตำบลขึ้นไปยังสามารถเก็บไว้ได้
- pass            : not sensitive — including PREFIX (คำนำหน้า/title like นาย/นาง/นางสาว/Mr/Mrs are public, not sensitive)

Thai language notes (common false positives):
- ถ้าเป็นบ้านเลขที่ จะมีต้อง / อย่างเช่น "เลขที่ 123/45" หรือ "บ้านเลขที่ 123 หมู่ 5" — ถ้าไม่มีตัวเลขหรือมีแค่เลขอย่างเดียว ให้พิจารณาว่าเป็น pass
- "ที่" alone = ordinal marker / row index (ที่ 1, ที่ 2) — NOT an address, decide pass
- "ชื่อ" in compound words like "ชื่อโรงเรียน", "ชื่อสถานที่", "ชื่อสสว." = name of a place/organization — NOT a person name, decide pass
- Always use sample values to confirm: numeric-only values (1, 2, 3) strongly indicate index/sequence columns
- ตำบล, แขวง, อำเภอ, เขต, จังหวัด, district, province = public geographic administrative units — NOT FULLNAME, NOT sensitive — decide pass
- postal_code, zip_code, รหัสไปรษณีย์ — these are public geographic data (same as province/district), decide pass
- birth_date, date_of_birth, วันเกิด, วันเดือนปีเกิด and similar date columns are NOT sensitive — decide pass
- FULLNAME applies ONLY when the column value is a person's actual name string (e.g. "สมชาย ใจดี"). Columns describing personal attributes are NOT FULLNAME — decide pass:
  * gender/เพศ, nationality/สัญชาติ/เชื้อชาติ, occupation/อาชีพ, religion/ศาสนา
  * disease/โรคประจำตัว, disability/ความพิการ, health condition/สุขภาพ
  * welfare program/สวัสดิการด้าน*, social problem/ปัญหาด้าน*, target group/กลุ่มเป้าหมาย
  * any column whose values are categories, codes, or yes/no flags — NOT a person name
  * organization/agency name: "หน่วยงาน", "องค์กร", "บริษัท" = name of a place/org — NOT a person name, decide pass
- สัญชาติ, สัญชาติอื่น, เชื้อชาติ, nationality, ethnicity — categorical value (Thai/ไทย, American, etc.) — NOT a person name, NOT PREFIX — decide pass
- "ปัญหาด้าน*" columns (ปัญหาด้านสุขภาพ, ปัญหาด้านครอบครัว, ปัญหาด้านความรุนแรง, etc.) — categorical yes/no or descriptive problem flags, NOT a person name — decide pass
- pipe-separated values (e.g. "เบาหวาน|ความดัน|โรคหัวใจ") = multi-select categorical field — NOT a person name, NOT FULLNAME — decide pass
- รหัสประจำบ้าน, house code, รหัสบ้าน = reference code/ID assigned to a house (not a street address field) — decide pass
- columns starting with "ลักษณะ" (ลักษณะที่อยู่อาศัย, ลักษณะครอบครัว, ลักษณะความพิการ, etc.) = characteristic/attribute type columns whose values are categories (e.g. เช่า/ซื้อ/อื่นๆ) — NOT an address field, NOT FULLNAME — decide pass
- "cm", "CM" in column names = Case Manager (social worker role code), NOT an email address — do NOT classify as EMAIL
- columns starting with "รหัส" (รหัส cm, รหัสครัวเรือน, รหัสประจำบ้าน, etc.) = reference code or ID field, NOT personal data — decide pass 


Respond with JSON only, no extra text:
{
  "sensitive_type": "<FULLNAME|PREFIX|FIRSTNAME|LASTNAME|EMAIL|ADDRESS_SHORT|ADDRESS_FULL|GEO|null>",
  "decision": "<masking|partial_masking|pass>",
  "confidence": <0.0 to 1.0>,
  "reason": "<brief explanation in English>"
}"""


def _build_llm_user_prompt(column_name: str, sample_values: List[str]) -> str:
    samples_str = ", ".join(f'"{v}"' for v in sample_values[:10]) if sample_values else "none"
    return f'Column name: "{column_name}"\nSample values: [{samples_str}]'


def _parse_llm_response(raw: str) -> Optional[Dict[str, Any]]:
    try:
        raw = raw.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except Exception:
        return None


class OpenAIProvider:
    """LLM provider using OpenAI API. Default provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        if openai is None:
            raise ImportError("pip install openai")
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def complete(self, column_name: str, sample_values: List[str]) -> Optional[Dict[str, Any]]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _build_llm_user_prompt(column_name, sample_values)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return _parse_llm_response(response.choices[0].message.content)


class OllamaProvider:
    """LLM provider using Ollama (local installation required)."""

    def __init__(self, model: str = "llama3.2"):
        if ollama_client is None:
            raise ImportError("pip install ollama")
        self.model = model

    def complete(self, column_name: str, sample_values: List[str]) -> Optional[Dict[str, Any]]:
        response = ollama_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _build_llm_user_prompt(column_name, sample_values)},
            ],
        )
        return _parse_llm_response(response["message"]["content"])


class ClaudeProvider:
    """LLM provider using Anthropic Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        if anthropic is None:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, column_name: str, sample_values: List[str]) -> Optional[Dict[str, Any]]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_LLM_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_llm_user_prompt(column_name, sample_values)},
            ],
        )
        return _parse_llm_response(response.content[0].text)


class HFLLMProvider:
    """LLM provider using HuggingFace Inference API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        if HFInferenceClient is None:
            raise ImportError("pip install huggingface_hub")
        self._client = HFInferenceClient(api_key=api_key or os.getenv("HF_TOKEN"))
        self.model = model

    def complete(self, column_name: str, sample_values: List[str]) -> Optional[Dict[str, Any]]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _build_llm_user_prompt(column_name, sample_values)},
            ],
            temperature=0,
            max_tokens=256,
        )
        return _parse_llm_response(response.choices[0].message.content)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class SensitiveColumnClassifier:
    def __init__(
        self,
        fuzzy_threshold: float = 92.0,
        semantic_threshold: float = 0.85,
        llm_threshold: float = 0.7,
        semantic_provider=None,
        llm_provider=None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            fuzzy_threshold:    minimum fuzzy score (0-100) to classify as sensitive
            semantic_threshold: minimum cosine similarity (0-1) to classify as sensitive
            llm_threshold:      minimum LLM confidence (0-1) to classify as sensitive
            semantic_provider:  LocalSemanticProvider | HFSemanticProvider | None (uses LocalSemanticProvider with embedding_model)
            llm_provider:       OpenAIProvider | OllamaProvider | ClaudeProvider | HFLLMProvider | None (uses OllamaProvider with llama3.2)
            embedding_model:    model name for LocalSemanticProvider when semantic_provider is None
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.semantic_provider = semantic_provider if semantic_provider is not None else LocalSemanticProvider(embedding_model)
        self.llm_provider = llm_provider if llm_provider is not None else OllamaProvider()

        # normalize term lists once at init
        self._categories: Dict[str, Dict[str, Any]] = {
            cat: {
                "terms": [_normalize(t) for t in data["terms"]],
                "decision": data["decision"],
            }
            for cat, data in _SENSITIVE_CATEGORIES.items()
        }

    def classify(self, column: ColumnInput) -> SensitiveClassificationResult:
        name = _normalize(column.column_name)
        result = SensitiveClassificationResult(
            column_name=column.column_name,
            normalized_name=name,
        )

        # ── Stage 1: Exact ────────────────────────────────────────────────────
        for cat, data in self._categories.items():
            exact, exact_term = _exact_match(name, data["terms"])
            if exact:
                result.lexical_exact = True
                result.lexical_exact_term = exact_term
                result.sensitive_type = cat
                result.decision = data["decision"]
                result.reason = "lexical_exact_match"
                result.confidence = 1.0
                result.metadata["matched_stage"] = "exact"
                return result

        # ── Stage 2: Fuzzy ────────────────────────────────────────────────────
        scores_per_cat: Dict[str, Tuple[float, Optional[str]]] = {
            cat: _fuzzy_match(name, data["terms"])
            for cat, data in self._categories.items()
        }

        sorted_cats = sorted(scores_per_cat, key=lambda c: scores_per_cat[c][0], reverse=True)
        best_cat = sorted_cats[0]
        best_score, best_term = scores_per_cat[best_cat]
        runner_up_score = scores_per_cat[sorted_cats[1]][0] if len(sorted_cats) > 1 else 0.0

        result.lexical_fuzzy_score = best_score
        result.lexical_fuzzy_term = best_term

        # ถ้า runner-up ก็ผ่าน threshold และห่างจาก best ไม่เกิน 5 คะแนน
        # แสดงว่า fuzzy ไม่มั่นใจพอ — ปล่อยให้ semantic/LLM จัดการแทน
        fuzzy_clear_winner = (
            runner_up_score < self.fuzzy_threshold
            or abs(best_score - runner_up_score) > 20
        )
        
        if best_score >= self.fuzzy_threshold and best_cat and fuzzy_clear_winner:
            result.sensitive_type = best_cat
            result.decision = self._categories[best_cat]["decision"]
            result.reason = "lexical_fuzzy_match"
            result.confidence = best_score / 100.0
            result.metadata["matched_stage"] = "fuzzy"
            return result

        # ── Stage 3: Semantic ─────────────────────────────────────────────────
        if self.semantic_provider is not None:
            sem_score, sem_cat = self.semantic_provider.score(name)
            result.semantic_score = sem_score
            result.semantic_category = sem_cat

            if sem_score >= self.semantic_threshold:
                result.sensitive_type = sem_cat
                result.decision = self._categories[sem_cat]["decision"]
                result.reason = "semantic_match"
                result.confidence = sem_score
                result.metadata["matched_stage"] = "semantic"
                return result

        # ── Stage 4: LLM ──────────────────────────────────────────────────────
        if self.llm_provider is not None:
            llm_result = self.llm_provider.complete(column.column_name, column.sample_values)
            if llm_result:
                llm_type = llm_result.get("sensitive_type")
                llm_decision = llm_result.get("decision", "pass")
                llm_confidence = float(llm_result.get("confidence", 0.0))
                llm_reason = llm_result.get("reason", "")

                if llm_decision in ("masking", "partial_masking") and llm_type and llm_confidence >= self.llm_threshold:
                    # ใช้ canonical decision จาก term list เสมอ ถ้า type นั้นรู้จัก
                    # LLM ระบุ type ได้ดี แต่ decision ให้ใช้ของเราเป็น source of truth
                    canonical_decision = self._categories.get(llm_type, {}).get("decision")
                    effective_decision = canonical_decision if canonical_decision is not None else llm_decision
                    result.sensitive_type = llm_type if effective_decision != "pass" else None
                    result.decision = effective_decision
                    result.reason = f"llm_match: {llm_reason}"
                    result.confidence = llm_confidence
                    result.metadata["matched_stage"] = "llm"
                    result.metadata["llm_raw"] = llm_result
                    return result

        # ── Value Pattern: Email guardrail ────────────────────────────────────
        if _has_email_pattern(column.sample_values):
            result.sensitive_type = "EMAIL"
            result.value_pattern_signal = True
            result.decision = "masking"
            result.reason = "value_pattern_email"
            result.confidence = 1.0
            result.metadata["matched_stage"] = "value_pattern"
            return result

        # ── Value Pattern: Full address guardrail ─────────────────────────────
        if _has_full_address_pattern(column.sample_values):
            result.sensitive_type = "ADDRESS_FULL"
            result.value_pattern_signal = True
            result.decision = "partial_masking"
            result.reason = "value_pattern_full_address"
            result.confidence = 1.0
            result.metadata["matched_stage"] = "value_pattern"
            return result

        # ── pass ──────────────────────────────────────────────────────────────
        result.decision = "pass"
        result.reason = "not_sensitive"
        result.confidence = result.semantic_score or 0.0
        return result
