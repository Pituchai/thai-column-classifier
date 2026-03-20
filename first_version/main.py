import csv
from column_classifier import ColumnClassifier, ColumnInput

clf = ColumnClassifier()

# โหลด expected
with open("data/expected.csv", encoding="utf-8-sig") as f:
    expected = {row["column_name"]: row["expected_decision"] for row in csv.DictReader(f)}

# โหลด test_data และ classify แต่ละ column
passed, passed_cases, failed = 0, [], []

with open("data/test_data.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

for col_name in fieldnames:
    samples = []
    for row in rows:
        value = row.get(col_name) or ""
        value = value.strip()
        if value:
            samples.append(value)
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
