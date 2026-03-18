import csv
from column_classifier import ColumnClassifier, ColumnInput

clf = ColumnClassifier()
output_rows = []

# โหลด expected
with open("expected.csv") as f:
    expected = {row["column_name"]: row["expected_decision"] for row in csv.DictReader(f)}

# โหลด test_data และ classify แต่ละ column
passed, failed = 0, []

with open("test_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

missing_expected = [col_name for col_name in fieldnames if col_name not in expected]
if missing_expected:
    missing_cols = ", ".join(missing_expected)
    raise ValueError(
        "expected.csv is missing expected_decision rows for these columns: "
        f"{missing_cols}"
    )

for col_name in fieldnames:
    samples = [row[col_name] for row in rows if row[col_name].strip()]
    result = clf.classify(ColumnInput(column_name=col_name, sample_values=samples))
    expected_decision = expected[col_name]
    passed_case = result.decision == expected_decision

    output_rows.append({
        "column_name": result.column_name,
        "normalized_name": result.normalized_name,
        "expected_decision": expected_decision,
        "predicted_decision": result.decision,
        "passed": passed_case,
        "reason": result.reason,
        "confidence": round(result.confidence, 6),
        "lexical_exact": result.lexical_exact,
        "lexical_exact_term": result.lexical_exact_term or "",
        "lexical_fuzzy_score": round(result.lexical_fuzzy_score, 6),
        "lexical_fuzzy_term": result.lexical_fuzzy_term or "",
        "semantic_score": round(result.semantic_score, 6) if result.semantic_score is not None else "",
        "semantic_term": result.semantic_term or "",
        "generic_name_risk": result.generic_name_risk,
        "value_pattern_signal": result.value_pattern_signal,
        "sample_values_count": len(samples),
        "matched_stage": result.metadata.get("matched_stage", ""),
    })

    if passed_case:
        passed += 1
    else:
        failed.append({
            "column":   col_name,
            "expected": expected_decision,
            "got":      result.decision,
            "reason":   result.reason,
            "confidence": round(result.confidence, 3),
        })

with open("classification_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "column_name",
            "normalized_name",
            "expected_decision",
            "predicted_decision",
            "passed",
            "reason",
            "confidence",
            "lexical_exact",
            "lexical_exact_term",
            "lexical_fuzzy_score",
            "lexical_fuzzy_term",
            "semantic_score",
            "semantic_term",
            "generic_name_risk",
            "value_pattern_signal",
            "sample_values_count",
            "matched_stage",
        ],
    )
    writer.writeheader()
    writer.writerows(output_rows)

# Summary
print(f"\n{'='*60}")
print(f"RESULT : {passed}/{passed + len(failed)} passed")
print(f"{'='*60}")
print("Saved detailed results to classification_results.csv")

if failed:
    print("\nFAILED CASES:")
    for f in failed:
        print(f"  ✗ {f['column']:<30} expected={f['expected']:<15} got={f['got']:<15} reason={f['reason']} conf={f['confidence']}")
else:
    print("\n✓ All test cases passed!")
