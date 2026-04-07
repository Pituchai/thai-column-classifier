import os
import pandas as pd
from src.thai_id_column_detector import ColumnClassifier, ColumnInput

clf = ColumnClassifier()


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, on_bad_lines="skip", engine="python")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# โหลด expected
expected_df = load_file("data/test_csv_1/expected.csv")
expected = dict(zip(expected_df["column_name"], expected_df["expected_decision"]))

# โหลด test_data และ classify แต่ละ column
passed, passed_cases, failed = 0, [], []

df = load_file("data/test_csv_1/test_data.csv")
fieldnames = df.columns.tolist()

for col_name in fieldnames:
    samples = df[col_name].dropna().astype(str).str.strip()
    samples = samples[samples != ""].tolist()
    result = clf.classify(ColumnInput(column_name=col_name, sample_values=samples))

    if result.decision == expected[col_name]:
        passed += 1
        passed_cases.append({
            "column": col_name,
            "decision": result.decision,
            "reason": result.reason,
            "confidence": round(result.confidence, 3),
        })
    else:
        failed.append({
            "column": col_name,
            "expected": expected[col_name],
            "got": result.decision,
            "reason": result.reason,
            "confidence": round(result.confidence, 3),
        })

# Summary
print(f"\n{'='*60}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'='*60}")

if passed_cases:
    print("\nPASSED CASES:")
    for case in passed_cases:
        print(f"  ✓ {case['column']:<30} decision={case['decision']:<15} reason={case['reason']} conf={case['confidence']}")

if failed:
    print("\nFAILED CASES:")
    for f in failed:
        print(f"  ✗ {f['column']:<30} expected={f['expected']:<15} got={f['got']:<15} reason={f['reason']} conf={f['confidence']}")
else:
    print("\n✓ All test cases passed!")
