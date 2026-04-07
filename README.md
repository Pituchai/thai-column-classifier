# Thai Column Classifier

A Python library for detecting and classifying sensitive columns in Thai datasets, designed to support PDPA (Personal Data Protection Act) compliance.

## Features

- **CID detection** — identifies Thai national ID (`เลขบัตรประชาชน`) columns via exact match, fuzzy match, semantic similarity, and LLM fallback
- **Sensitive column detection** — classifies columns as `FULLNAME`, `PREFIX`, `FIRSTNAME`, `LASTNAME`, `EMAIL`, `ADDRESS_SHORT`, or `ADDRESS_FULL`
- **Pluggable LLM providers** — OpenAI, Claude, Ollama, HuggingFace
- **Pluggable semantic providers** — local `sentence-transformers` or HuggingFace Inference API

## Detection pipeline

Each column goes through up to 4 stages (stops early if confident):

1. **Exact match** — keyword list
2. **Fuzzy match** — `rapidfuzz` ratio / partial_ratio / token_sort_ratio
3. **Semantic** — embedding cosine similarity
4. **LLM** — prompt-based classification as final fallback

## Output decisions

| Detector | Decision | Meaning |
|---|---|---|
| CID | `auto_hash` | Column is a Thai national ID — hash it |
| CID | `human_review` | Likely a CID — needs manual review |
| Sensitive | `masking` | Mask the entire value |
| Sensitive | `partial_masking` | Mask street-level part only |
| Both | `pass` | Not sensitive |

## Installation

```bash
pip install git+https://github.com/<your-username>/column_classification_poc.git
```

### Dependencies

```bash
pip install -r requirements.txt
```

Optional — install only the providers you need:

```bash
pip install openai        # OpenAI LLM provider
pip install anthropic     # Claude LLM provider
pip install ollama        # Ollama LLM provider
pip install sentence-transformers  # local semantic provider
```

## Quick start

```python
from src.thai_id_column_detector import ColumnClassifier, ColumnInput
from src.thai_sensitive_column_detector import SensitiveColumnClassifier, ColumnInput as SensitiveInput, OpenAIProvider

# CID detector
cid_clf = ColumnClassifier()
result = cid_clf.classify(ColumnInput(
    column_name="เลขบัตรประชาชน",
    sample_values=["1101700203451"]
))
print(result.decision)  # auto_hash

# Sensitive column detector
sensitive_clf = SensitiveColumnClassifier(llm_provider=OpenAIProvider())
result = sensitive_clf.classify(SensitiveInput(
    column_name="ชื่อ-นามสกุล",
    sample_values=["สมชาย ใจดี"]
))
print(result.decision)  # masking
```

## Environment variables

```bash
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...        # for HuggingFace providers
```

## Running tests

```bash
python test_check_id.py
python test_check_sensitive.py
```

## Project structure

```
.
├── src/
│   ├── thai_id_column_detector.py       # CID classifier
│   └── thai_sensitive_column_detector.py # Sensitive column classifier
├── data/                                 # Test datasets
├── main.py                               # Example runner
├── test_check_id.py
└── test_check_sensitive.py
```
