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

- `column_classifier.py` contains the reusable classifier library
- `main.py` runs the included CSV-based evaluation
- `test_data.csv` contains sample column data
- `expected.csv` defines the expected decision for each test column
- `classification_results.csv` is the generated evaluation output

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

Run the provided evaluation script:

```bash
python3 main.py
```

The script will:

- load expected outcomes from `expected.csv`
- classify every column in `test_data.csv`
- compare predicted decisions with expected decisions
- write detailed results to `classification_results.csv`

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

- The current heuristic is focused on Thai citizen ID style fields and related English aliases
- Generic names such as `id` may be routed to `human_review` instead of `auto_hash`
- Sample values help the guardrail detect likely 13-digit identifier patterns
