# Column Classification POC

Proof of concept for classifying dataset columns that may contain Thai citizen ID data and routing them to one of three decisions:

- `auto_hash`
- `human_review`
- `pass`

The classifier combines:

- exact lexical matching
- fuzzy lexical matching
- optional semantic similarity
- simple value-pattern guardrails for 13-digit identifiers

## Project Files

- `classifier/engine.py` core classification pipeline
- `classifier/models.py` input/output/config dataclasses
- `classifier/normalize.py` text normalization logic
- `classifier/registry.py` YAML registry loader
- `classifier/registry/cid_registry.yaml` CID term registry (CID terms, generic terms, semantic references, replacements)
- `column_classifier.py` backward-compatible public import entrypoint
- `main.py` CSV-based evaluation runner
- `data/test_data.csv` sample column data
- `data/expected.csv` expected decision per test column

- `first_version/` legacy snapshot for presentation (single-file version)

## Requirements

- Python 3.10+
- packages from `requirements.txt`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional packages for semantic matching:

```bash
python3 -m pip install sentence-transformers python-dotenv
```

If semantic packages are not installed, the classifier falls back to lexical-only behavior when possible.

## Run The Demo

```bash
python3 main.py
```

The script will:

- load expected outcomes from `data/expected.csv`
- classify every column in `data/test_data.csv`
- compare predicted decisions with expected decisions
- print failed cases and summary to terminal

## Library Usage

```python
from column_classifier import ColumnClassifier, ColumnInput

classifier = ColumnClassifier()

result = classifier.classify(
    ColumnInput(
        column_name="เลขบัตรประชาชน",
        sample_values=["1101700203451", "3101200198765"],
    )
)

print(result.decision)
print(result.reason)
print(result.confidence)
```

## Decisions

- `auto_hash`: high-confidence sensitive column, safe to process automatically
- `human_review`: uncertain or generic column name that should be checked manually
- `pass`: not classified as a citizen ID column

## Semantic Backend Options

`ClassifierConfig.semantic_backend` supports:

- `auto` tries local embeddings first, then Hugging Face API if configured
- `local` requires `sentence-transformers`
- `hf_api` requires `huggingface_hub` plus `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`
- `disabled` turns off semantic scoring

Example:

```python
from column_classifier import ColumnClassifier, ClassifierConfig

config = ClassifierConfig(
    semantic_backend="disabled",
)

classifier = ColumnClassifier(config=config)
```

## Notes

- The CID terms are now stored in `classifier/registry/cid_registry.yaml`
- You can override the registry path with `ClassifierConfig(registry_path="...")`
- Generic names such as `id` may be routed to `human_review` instead of `auto_hash`
- Sample values help the guardrail detect likely 13-digit identifier patterns
