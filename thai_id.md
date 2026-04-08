# Thai ID Column Detector

ตรวจจับคอลัมน์เลขบัตรประชาชนไทย (Thai national ID / CID)

## ผลลัพธ์

- `auto_hash` — มั่นใจสูง ควร hash อัตโนมัติ
- `pass` — ไม่ใช่คอลัมน์เลขบัตรประชาชน

## การใช้งาน

```python
from thai_column_classifier import IDColumnClassifier, IDColumnInput

clf = IDColumnClassifier()
result = clf.classify(IDColumnInput(
    column_name="เลขบัตรประชาชน",
    sample_values=["1101700203451", "3101200198765"],
))

print(result.decision)    # auto_hash
print(result.reason)      # lexical_exact_match
print(result.confidence)  # 1.0
```

## Detection Pipeline

```
normalize()
    │
    ├─ Stage 1: Exact Match (_CID_TERMS)           → auto_hash
    │
    ├─ Stage 2: Fuzzy Match
    │   ├─ score ≥ 92                              → auto_hash
    │   └─ score < 92                              → Stage 3
    │
    ├─ Stage 3: Semantic Score
    │   ├─ score ≥ 0.93                            → auto_hash
    │   └─ score < 0.93                            → Stage 4
    │
    └─ Stage 4: 13-digit Guardrail
        ├─ ≥ 25% of non-null values = 13 digits    → auto_hash (confidence = 1.0)
        └─ otherwise                               → pass
```

## Semantic Backend

`ClassifierConfig.semantic_backend` รองรับค่าเหล่านี้ (ค่าเริ่มต้นคือ `local`):

- `local` — ใช้ `sentence-transformers` รันบนเครื่อง (ค่าเริ่มต้น)
- `hf_api` — ใช้ `huggingface_hub` และต้องมี `HF_TOKEN`
- `auto` — ลองใช้ local model ก่อน ถ้าไม่ได้ค่อยลอง HuggingFace API
- `disabled` — ปิด semantic scoring

```python
from thai_column_classifier import IDColumnClassifier, ClassifierConfig

clf = IDColumnClassifier(config=ClassifierConfig(semantic_backend="disabled"))
```

## Threshold Reference

| Threshold | ค่า | หมายเหตุ |
|-----------|-----|---------|
| `fuzzy_auto_threshold` | 92.0 | 90+ = precision critical use case |
| `semantic_auto_threshold` | 0.93 | heuristic |
| `min_ratio` (13-digit) | 0.25 | heuristic |

## 13-digit Guardrail

- นับเฉพาะ non-null values (`none`, `nan`, `null`, `na`, `n/a`, `nat`, `<na>` ถูก exclude)
- ต้องเป็นตัวเลขล้วน 13 หลักพอดี และผ่าน Thai CID checksum
- threshold เริ่มต้น 25%

## ตัวอย่างการ normalize

| Input | Output |
|-------|--------|
| `citizenid` | `citizen id` |
| `nationalid` | `national id` |
| `idcard` | `id card` |
| `เลขบตรประชาชน` | `เลขบัตรประชาชน` |
| `เลขบัตรปชช` | `เลขบัตรประชาชน` |
