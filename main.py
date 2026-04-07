import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# both id and sensitive detectors need to be run in the same script to ensure the correct order of execution and to allow the sensitive detector to only run on columns that pass the CID check.
from src.thai_id_column_detector import ColumnClassifier as CIDClassifier, ColumnInput as CIDInput
from src.thai_sensitive_column_detector import SensitiveColumnClassifier, ColumnInput as SensitiveInput, OpenAIProvider

cid_clf = CIDClassifier()
sensitive_clf = SensitiveColumnClassifier(
    llm_provider=OpenAIProvider(),
)


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, on_bad_lines="skip", engine="python")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def classify_column(col_name: str, samples: list) -> dict:
    # CID detector runs first
    cid_result = cid_clf.classify(CIDInput(column_name=col_name, sample_values=samples))
    if cid_result.decision != "pass":
        return {
            "decision": cid_result.decision,
            "type": "CID",
            "reason": cid_result.reason,
            "confidence": round(cid_result.confidence, 3),
            "detector": "cid",
        }

    # Sensitive detector runs if CID says pass
    sensitive_result = sensitive_clf.classify(SensitiveInput(column_name=col_name, sample_values=samples))
    return {
        "decision": sensitive_result.decision,
        "type": sensitive_result.sensitive_type,
        "reason": sensitive_result.reason,
        "confidence": round(sensitive_result.confidence, 3),
        "detector": "sensitive",
    }


# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/test_csv_5/test_data.csv"
EXPECTED_PATH = "data/test_csv_5/expected.csv"

expected_df = load_file(EXPECTED_PATH)
expected = dict(zip(expected_df["column_name"], expected_df["expected_decision"]))

df = load_file(DATA_PATH)

passed, passed_cases, failed = 0, [], []

for col_name in df.columns.tolist():
    samples = df[col_name].dropna().astype(str).str.strip()
    samples = samples[samples != ""].tolist()

    if col_name not in expected:
        continue

    result = classify_column(col_name, samples)

    if result["decision"] == expected[col_name]:
        passed += 1
        passed_cases.append({"column": col_name, **result})
    else:
        failed.append({"column": col_name, "expected": expected[col_name], **result})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'='*70}")

if passed_cases:
    print("\nPASSED:")
    for c in passed_cases:
        print(f"  ✓ {c['column']:<35} decision={c['decision']:<15} type={str(c['type']):<12} reason={c['reason']:<30} detector={c['detector']} conf={c['confidence']}")

if failed:
    print("\nFAILED:")
    for f in failed:
        print(f"  ✗ {f['column']:<35} expected={f['expected']:<15} got={f['decision']:<15} type={str(f['type']):<12} reason={f['reason']:<30} detector={f['detector']} conf={f['confidence']}")
else:
    print("\n✓ All test cases passed!")
