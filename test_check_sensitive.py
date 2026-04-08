import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from thai_column_classifier import SensitiveColumnClassifier, SensitiveColumnInput as ColumnInput, OllamaProvider

clf = SensitiveColumnClassifier(
    llm_provider=OllamaProvider(model="llama3.2"),
)


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, on_bad_lines="skip", engine="python")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


expected_df = load_file("data/test_csv_4/expected.csv")
expected = dict(zip(expected_df["column_name"], expected_df["expected_decision"]))

passed, passed_cases, failed = 0, [], []

df = load_file("data/test_csv_4/test_data.csv")

for col_name in df.columns.tolist():
    samples = df[col_name].dropna().astype(str).str.strip()
    samples = samples[samples != ""].tolist()
    result = clf.classify(ColumnInput(column_name=col_name, sample_values=samples))

    if col_name not in expected:
        continue

    if result.decision == expected[col_name]:
        passed += 1
        passed_cases.append({
            "column": col_name,
            "decision": result.decision,
            "type": result.sensitive_type,
            "stage": result.metadata.get("matched_stage", "-"),
            "confidence": round(result.confidence, 3),
        })
    else:
        failed.append({
            "column": col_name,
            "expected": expected[col_name],
            "got": result.decision,
            "type": result.sensitive_type,
            "stage": result.metadata.get("matched_stage", "-"),
            "confidence": round(result.confidence, 3),
        })

# Summary
print(f"\n{'='*70}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'='*70}")

if passed_cases:
    print("\nPASSED:")
    for c in passed_cases:
        print(f"  ✓ {c['column']:<35} decision={c['decision']:<15} type={str(c['type']):<12} stage={c['stage']} conf={c['confidence']}")

if failed:
    print("\nFAILED:")
    for f in failed:
        print(f"  ✗ {f['column']:<35} expected={f['expected']:<15} got={f['got']:<15} type={str(f['type']):<12} stage={f['stage']} conf={f['confidence']}")
else:
    print("\n✓ All test cases passed!")
