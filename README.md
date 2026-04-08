# thai-column-classifier

A Python library for detecting and classifying sensitive columns in Thai datasets, designed to support PDPA (Personal Data Protection Act) compliance.

## Features

- **CID detection** — identifies Thai national ID (`เลขบัตรประชาชน`) columns via exact match, fuzzy match, semantic similarity, and value pattern (13-digit checksum)
- **Sensitive column detection** — classifies columns as `FULLNAME`, `PREFIX`, `FIRSTNAME`, `LASTNAME`, `EMAIL`, `ADDRESS_SHORT`, or `ADDRESS_FULL`
- **Pluggable LLM providers** — Ollama (local), OpenAI, Claude, HuggingFace
- **Pluggable semantic providers** — local `sentence-transformers` or HuggingFace Inference API
- **Privacy-first by default** — all inference runs locally out of the box (Ollama + sentence-transformers); no data sent to external APIs

## Detection pipeline

Each column goes through stages in order, stopping early if confident enough.

**CID detector** (`เลขบัตรประชาชน`):
1. **Exact match** — keyword list
2. **Fuzzy match** — `rapidfuzz` ratio / partial_ratio / token_sort_ratio
3. **Semantic** — embedding cosine similarity
4. **Guardrail** — 13-digit checksum pattern on sample values

**Sensitive column detector**:
1. **Exact match** — keyword list per category
2. **Fuzzy match** — `rapidfuzz` ratio / partial_ratio / token_sort_ratio
3. **Semantic** — embedding cosine similarity
4. **LLM** — prompt-based classification as final fallback
5. **Guardrails** — email pattern and full address pattern on sample values

## Output decisions

| Detector | Decision | Meaning |
|---|---|---|
| CID | `auto_hash` | Column is a Thai national ID — hash it |
| Sensitive | `masking` | Mask the entire value (`****`) |
| Sensitive | `partial_masking` | Mask street-level part only, keep district/province |
| Both | `pass` | Not sensitive |

## Supported input formats

The classifier operates on a pandas `DataFrame` — it is file-format agnostic. The `load_file()` helper in `main.py` supports:

| Format | Extensions |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |

To support additional formats (Parquet, JSON, database queries, etc.), you only need to extend `load_file()`. The classifier itself works unchanged as long as you pass it a `DataFrame`.

## Installation

```bash
pip install thai-column-classifier
```

Install with optional providers:

```bash
pip install thai-column-classifier[ollama]     # Ollama (local LLM)
pip install thai-column-classifier[semantic]   # local sentence-transformers
pip install thai-column-classifier[openai]     # OpenAI
pip install thai-column-classifier[claude]     # Anthropic Claude
pip install thai-column-classifier[all]        # everything
```

## Quick start

### Classify a single column

```python
from thai_column_classifier import ThaiColumnClassifier, OllamaProvider

clf = ThaiColumnClassifier(llm_provider=OllamaProvider(model="llama3.2"))

result = clf.classify("เลขบัตรประชาชน", samples=["1101700203451"])
print(result.decision)  # auto_hash
print(result.type)      # CID
```

### Load a file and classify all columns

```python
import pandas as pd
from thai_column_classifier import ThaiColumnClassifier, OllamaProvider

clf = ThaiColumnClassifier(llm_provider=OllamaProvider(model="llama3.2"))

# Load CSV
df = pd.read_csv("your_data.csv", encoding="utf-8-sig", dtype=str)

# Load Excel
# df = pd.read_excel("your_data.xlsx", dtype=str)

results = clf.classify_dataframe(df)

for col_name, result in results.items():
    print(f"{col_name}: decision={result.decision}, type={result.type}")
```

### Using the built-in `load_file()` helper (CSV and Excel)

```python
import os
import pandas as pd
from thai_column_classifier import ThaiColumnClassifier, OllamaProvider

def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, on_bad_lines="skip", engine="python")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

clf = ThaiColumnClassifier(llm_provider=OllamaProvider(model="llama3.2"))

df = load_file("data/my_data.csv")
results = clf.classify_dataframe(df)

for col_name, result in results.items():
    print(f"{col_name:<30} decision={result.decision:<15} type={result.type}")
```

## Environment variables

```bash
HF_TOKEN=hf_...          # for HuggingFace providers
OPENAI_API_KEY=sk-...    # for OpenAI provider
ANTHROPIC_API_KEY=...    # for Claude provider
```

## Running tests

```bash
python test_check_id.py
python test_check_sensitive.py
python main.py
```

## Project structure

```
.
├── thai_column_classifier/
│   ├── __init__.py
│   ├── classifier.py                     # Unified ThaiColumnClassifier
│   ├── thai_id_column_detector.py        # CID classifier
│   └── thai_sensitive_column_detector.py # Sensitive column classifier
├── data/                                 # Test datasets
├── pyproject.toml
├── main.py
├── test_check_id.py
└── test_check_sensitive.py
```
