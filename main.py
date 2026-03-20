import csv

from column_classifier import ColumnClassifier, ColumnInput

clf = ColumnClassifier()

# โหลด expected
with open("data/expected.csv", encoding="utf-8") as f:
    expected = {row["column_name"]: row["expected_decision"] for row in csv.DictReader(f)}

# โหลด test_data และ classify แต่ละ column
passed, failed = 0, []

with open("data/test_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

for col_name in fieldnames:
    samples = [row[col_name] for row in rows if row[col_name].strip()]
    result = clf.classify(ColumnInput(column_name=col_name, sample_values=samples))

    if result.decision == expected[col_name]:
        passed += 1
    else:
        failed.append(
            {
                "column": col_name,
                "expected": expected[col_name],
                "got": result.decision,
                "reason": result.reason,
                "confidence": round(result.confidence, 3),
            }
        )

# Summary
print(f"\n{'=' * 60}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'=' * 60}")

if failed:
    print("\nFAILED CASES:")
    for f in failed:
        print(
            f"  ✗ {f['column']:<30} expected={f['expected']:<15} "
            f"got={f['got']:<15} reason={f['reason']} conf={f['confidence']}"
        )
else:
    print("\n✓ All test cases passed!")
