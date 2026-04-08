import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from thai_column_classifier import ThaiColumnClassifier, OllamaProvider

clf = ThaiColumnClassifier(llm_provider=OllamaProvider(model="llama3.2"))


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, on_bad_lines="skip", engine="python")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/test_csv_4/test_data.csv"
EXPECTED_PATH = "data/test_csv_4/expected.csv"

expected_df = load_file(EXPECTED_PATH)
expected = dict(zip(expected_df["column_name"], expected_df["expected_decision"]))

df = load_file(DATA_PATH)
results = clf.classify_dataframe(df)

passed, passed_cases, failed = 0, [], []

for col_name, result in results.items():
    if col_name not in expected:
        continue

    if result.decision == expected[col_name]:
        passed += 1
        passed_cases.append(result)
    else:
        failed.append({"expected": expected[col_name], "result": result})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'='*70}")

if passed_cases:
    print("\nPASSED:")
    for r in passed_cases:
        print(f"  ✓ {r.column_name:<35} decision={r.decision:<15} type={str(r.type):<12} reason={r.reason:<30} detector={r.detector} conf={r.confidence}")

if failed:
    print("\nFAILED:")
    for f in failed:
        r = f["result"]
        print(f"  ✗ {r.column_name:<35} expected={f['expected']:<15} got={r.decision:<15} type={str(r.type):<12} reason={r.reason:<30} detector={r.detector} conf={r.confidence}")
else:
    print("\n✓ All test cases passed!")
