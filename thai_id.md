# Column Classification POC

โปรเจกต์นี้เป็นตัวอย่างสำหรับใช้จำแนกชื่อคอลัมน์ว่าอาจเป็นข้อมูล `Thai citizen ID` หรือไม่ และส่งผลลัพธ์ออกมาเป็น 2 แบบ:

- `auto_hash`

- `pass`

ตัว classifier ใช้หลายวิธีร่วมกัน ได้แก่:

- exact lexical matching
- fuzzy lexical matching
- semantic similarity แบบ optional
- การดูรูปแบบค่าตัวอย่าง เช่นเลข 13 หลัก

## โครงสร้างไฟล์

- `column_classifier.py` เป็น logic หลักของ classifier
- `main.py` ใช้รันชุดทดสอบจากไฟล์ CSV
- `data/test_data.csv` เก็บข้อมูลตัวอย่างของแต่ละคอลัมน์
- `data/expected.csv` เก็บผลลัพธ์ที่คาดหวังของแต่ละคอลัมน์

## การติดตั้ง

แนะนำให้ใช้ virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ถ้าต้องการใช้งาน `.env` ให้ติดตั้งเพิ่ม:

```bash
pip install python-dotenv
```

ถ้าต้องการเปิด semantic matching แบบ local embedding ให้ติดตั้งเพิ่ม:

```bash
pip install sentence-transformers
```

## การตั้งค่า Hugging Face Token

ถ้าต้องการใช้ semantic backend ผ่าน Hugging Face API ให้ตั้งค่า `HF_TOKEN`

ตัวอย่างใน shell:

```bash
export HF_TOKEN="your_token_here"
```

ตัวอย่างในไฟล์ `.env`:

```env
HF_TOKEN="your_token_here"
```

หมายเหตุ:

- ห้ามเว้นวรรคหน้าและหลัง `=`
- ถ้าใช้ `.env` ต้องมี package `python-dotenv`

## การรัน

รันชุดทดสอบทั้งหมดด้วยคำสั่ง:

```bash
python main.py
```

ระบบจะ:

- โหลด expected result จาก `data/expected.csv`
- โหลด sample data จาก `data/test_data.csv`
- classify ทีละคอลัมน์
- เปรียบเทียบผลลัพธ์จริงกับ expected
- แสดง `PASSED CASES`, `FAILED CASES` และสรุปผล

## การใช้งานในโค้ด

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

## ความหมายของผลลัพธ์

- `auto_hash`: มั่นใจสูงว่าเป็นคอลัมน์ข้อมูลอ่อนไหว ควรจัดการ hash อัตโนมัติได้
- `pass`: ไม่มีสัญญาณว่าเป็นคอลัมน์เลขบัตรประชาชน

> หมายเหตุ: ไม่มี `human_review` อีกต่อไป — ระบบตัดสินใจเองทั้งหมดระหว่าง `auto_hash` และ `pass`

## หลักการตัดสินใจโดยสรุป

```
normalize()
    │
    ├─ Stage 1: Exact Match (_CID_TERMS)           → auto_hash
    │
    ├─ Stage 2: Fuzzy Match
    │   ├─ score ≥ 92                              → auto_hash
    │   └─ score < 92                              → continue to Stage 3
    │
    ├─ Stage 3: Semantic Score
    │   ├─ score ≥ 0.93                            → auto_hash
    │   └─ score < 0.93                            → continue to Stage 4
    │
    └─ Stage 4: 13-digit Guardrail
        ├─ ≥ 25% of non-null values = 13 digits    → auto_hash (confidence = 1.0)
        └─ otherwise                               → pass
```

หลักการ:

## Q&A: Stage 3 และ Stage 4

**Q: Stage 3 (semantic) กับ Stage 4 (13-digit) รันพร้อมกันไหม?**

A: ทั้งสองถูก **คำนวณพร้อมกันเสมอ** ก่อนที่จะตัดสินใจ — semantic score และ 13-digit signal จะถูก compute ทั้งคู่แล้วค่อยส่งเข้า `_decide()` พร้อมกัน ใน `_decide()` จะเช็ค Stage 3 ก่อน ถ้า semantic score ผ่าน threshold ก็ return ทันทีโดยไม่ดู Stage 4 แต่ถ้า semantic ไม่ผ่าน (หรือ backend ปิดอยู่) ค่อยดู Stage 4

**Q: แล้วถ้า semantic score อยู่กลางๆ (< 0.93) และมี 13-digit signal ด้วย ระบบจะเลือกอะไร?**

A: ถ้า semantic score < 0.93 จะ fall through ไป Stage 4 — 13-digit guardrail จะมีผลและอาจ return `auto_hash` ได้ ไม่มี `human_review` อีกต่อไป

**Ref: มองอีกแบบได้ว่า — "ต้องจบ Stage 3 ก่อนค่อยถึงคิวของ Stage 4"**

ถึงแม้ทั้งสองจะถูก compute พร้อมกันใน `classify()` แต่ใน `_decide()` Stage 3 มี priority เด็ดขาด ถ้า semantic score ≥ 0.93 จะ return `auto_hash` ทันที — Stage 4 ได้คิวก็ต่อเมื่อ Stage 3 "หมดสิทธิ์" ทั้งหมดแล้วเท่านั้น

---

## ตัวอย่างการ normalize

ตัวอย่างเช่น:

- `citizenid` -> `citizen id`
- `nationalid` -> `national id`
- `idcard` -> `id card`
- `เลขบตรประชาชน` -> `เลขบัตรประชาชน`
- `เลขบัตรปชช` -> `เลขบัตรประชาชน`

จุดประสงค์คือทำให้ชื่อที่พิมพ์ต่างรูปแบบกันสามารถ match กับ rule เดียวกันได้

## 13-digit Guardrail

- นับเฉพาะ non-null values (`none`, `nan`, `null`, `na`, `n/a`, `nat`, `<na>` ถูก exclude)
- ต้องเป็นตัวเลขล้วน 13 หลักพอดี (`re.fullmatch(r"\d{13}", ...)`) และผ่าน Thai CID checksum
- threshold เริ่มต้น 25% — ปรับได้ผ่าน `_has_13_digit_pattern(samples, min_ratio=...)`

**Known limitation — float type:**
ถ้า pandas โหลดคอลัมน์แล้ว infer เป็น `float64` ค่าจะกลายเป็น scientific notation เช่น `1.10170020345e+12` ซึ่งจะไม่ผ่าน `\d{13}` และ checksum จะไม่ถูกเรียก ยังไม่ได้จัดการ case นี้

## Semantic Backend

`ClassifierConfig.semantic_backend` รองรับค่าเหล่านี้:

- `auto` ลองใช้ local model ก่อน ถ้าไม่ได้ค่อยลอง Hugging Face API
- `local` ใช้ `sentence-transformers`
- `hf_api` ใช้ `huggingface_hub` และต้องมี `HF_TOKEN`
- `disabled` ปิด semantic scoring

ตัวอย่าง:

```python
from column_classifier import ColumnClassifier, ClassifierConfig

config = ClassifierConfig(
    semantic_backend="disabled",
)

classifier = ColumnClassifier(config=config)
```

## วิธีเช็คว่า semantic ทำงานหรือไม่

ใช้คำสั่งนี้:

```bash
python - <<'PY'
from column_classifier import ColumnClassifier
clf = ColumnClassifier()
print("semantic_backend:", clf._semantic_backend)
print("has_ref_embeddings:", clf._ref_embeddings is not None)
PY
```

ถ้า semantic backend ทำงานผ่าน Hugging Face API จะเห็นประมาณนี้:

```text
semantic_backend: hf_api
has_ref_embeddings: True
```

ถ้า semantic ถูกปิด จะเห็น:

```text
semantic_backend: disabled
has_ref_embeddings: False
```

## หมายเหตุเพิ่มเติม

- rule ชุดนี้เน้นไปที่การจับคอลัมน์ที่เกี่ยวกับเลขบัตรประชาชนไทย
- ถ้า semantic backend ยังไม่พร้อม ระบบจะ fallback ไปใช้ lexical logic และ 13-digit guardrail
- `_REPLACEMENTS` ใช้แปลงคำสะกดติดกัน คำย่อ หรือ typo ให้เป็นรูปแบบมาตรฐานก่อน match
- fuzzy match ที่ score < 92 จะไม่ return `human_review` อีกต่อไป แต่จะ fall through ไป Stage 3 (semantic) และ Stage 4 (13-digit) แทน
- 13-digit guardrail ใช้ `confidence = 1.0`
- remove human review, 
- add check thai id (logic check sum)
- llm ollama check context local, ของตอนทำ masking
- ไปดูมาว่าพวกค่า threshold ใช้อะไรในการตัดสินใจ เป็นค่า threshold ที่มาจาก default หรือเปล่า

## Threshold Reference

ค่า threshold ทั้งหมดใน `ClassifierConfig` เป็น **informed heuristic** ที่อิงจาก community best practice ไม่มี paper รับรองค่าตายตัว

| Threshold | ค่า | อ้างอิง |
|-----------|-----|---------|
| `fuzzy_auto_threshold` | 92.0 | DataCamp (Jan 2026): "90+ = precision critical use case" |
| `semantic_auto_threshold` | 0.93 | heuristic — ยังไม่มี ref จาก labeled data |
| `min_ratio` (13-digit) | 0.25 | heuristic — ยังไม่มี ref จาก labeled data |

> `fuzzy_review_threshold` และ `semantic_review_threshold` ถูกลบออกแล้ว เนื่องจากไม่มี `human_review` อีกต่อไป

**สรุปจาก community:**
- 90+ = เชื่อได้เลย (precision critical) → `auto_hash`
- ต่ำกว่า 90 = fall through ไป stage ถัดไป

> "Thresholds must reflect data quality and business risk — fixed thresholds across all datasets is an anti-pattern." — Data Ladder

**ข้อสังเกตสำหรับ project นี้:**
- domain แคบ (ชื่อคอลัมน์เลขบัตรประชาชนไทย) ทำให้ค่าเหล่านี้ใช้งานได้ดีสำหรับ POC
- risk จริงคือ false negative (พลาด CID column) มากกว่า false positive เพราะมี 13-digit checksum เป็น safety net ท้ายสุด
- ถ้าจะ production ควร tune ด้วย labeled dataset ของตัวเองก่อน