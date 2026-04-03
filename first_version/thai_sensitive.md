# Column Classification POC — Project Overview

โปรเจกต์นี้ตรวจจับและจำแนกคอลัมน์ข้อมูลอ่อนไหวในไฟล์ CSV/Excel โดยเน้นบริบทข้อมูลไทย

## โครงสร้างไฟล์

```
first_version/
├── src/
│   ├── thai_id_column_detector.py        # ตรวจจับคอลัมน์เลขบัตรประชาชนไทย
│   └── thai_sensitive_column_detector.py # ตรวจจับคอลัมน์ข้อมูลอ่อนไหวทั่วไป (อยู่ระหว่างพัฒนา)
├── main.py                               # รันชุดทดสอบ
├── requirements.txt
└── data/
    ├── test_csv_1/
    │   ├── test_data.csv
    │   └── expected.csv
    └── test_csv_2/
```

## Detectors

### thai_id_column_detector.py
ตรวจจับคอลัมน์เลขบัตรประชาชนไทย (Thai national ID / CID)

ผลลัพธ์:
- `auto_hash` — มั่นใจสูง ควร hash อัตโนมัติ

- `pass` — ไม่ใช่คอลัมน์เลขบัตรประชาชน

### thai_sensitive_column_detector.py
ตรวจจับคอลัมน์ข้อมูลอ่อนไหวทั่วไป ได้แก่ ชื่อ, นามสกุล, อีเมล, ที่อยู่, โซเชียลมีเดีย

ผลลัพธ์:
- `masking` — ข้อมูลละเอียดอ่อน ควร mask ทั้งหมด
- `partial_masking` — ข้อมูลละเอียดอ่อนบางส่วน เช่น ที่อยู่ยาว mask เฉพาะระดับต่ำกว่าตำบล
- `pass` — ไม่ใช่ข้อมูลละเอียดอ่อน

> อยู่ระหว่างพัฒนา

---

## Sensitive Column Types

| ลำดับ | กลุ่ม | Sensitive Type | Output |
|---|---|---|---|
| 1 | ชื่อ | FULLNAME, PREFIX, FIRSTNAME, LASTNAME | masking |
| 2 | อีเมล | EMAIL | masking |
| 3 | ที่อยู่สั้น | บ้านเลขที่, หมู่ที่, ซอย, ถนน ฯลฯ | masking |
| | | ตำบล, อำเภอ, จังหวัด ฯลฯ | pass |
| 4 | ที่อยู่ยาว | FULL_ADDRESS | partial_masking |
| 5 | โซเชียลมีเดีย | LINE, FACEBOOK, INSTAGRAM ฯลฯ | masking |

### ตัวอย่าง alias แต่ละกลุ่ม

**FULLNAME:** ชื่อ-นามสกุล, ชื่อ-สกุล, ชื่อสกุล, PERSON_NAME, RGT_NAME, FullName, NAME, NAMETHAI

**PREFIX:** คำนำหน้า, คำนำหน้านาม, คำนำหน้าชื่อ, TITLE_ID, TITLE_DESC, Prefix, title

**FIRSTNAME:** ชื่อ, ชื่อต้น, FNAME, FIRST_NAME, GiveName, encrypted_name, PersonFirstNameTH

**LASTNAME:** นามสกุล, สกุล, LNAME, LAST_NAME, FamilyName, encrypted_lastname, PersonLastNameTH

### ตัวอย่าง masking

```
masking         → ****
partial_masking → **** แขวงบางบอน เขตบางบอน กรุงเทพฯ 10150
```

---

## Detection Pipeline

ทุก column ผ่าน pipeline นี้ตามลำดับ หยุดทันทีที่ stage ใด stage หนึ่งตัดสินได้

```
column_name + sample_values
        ↓
┌─────────────────────────┐
│  Stage 1: Exact Match   │ → ถ้าเจอ: return ทันที (confidence = 1.0)
└─────────────────────────┘
        ↓ ไม่เจอ
┌─────────────────────────┐
│  Stage 2: Fuzzy Match   │
│  - ratio                │ → ถ้าเกิน threshold: return
│  - partial ratio        │
│    (name > term เท่านั้น)│
│  - token sort ratio     │
│    (English)            │
└─────────────────────────┘
        ↓ ไม่เจอ
┌─────────────────────────┐
│  Stage 3: Semantic      │ → ถ้าเกิน threshold: return
│  (SentenceTransformer)  │
└─────────────────────────┘
        ↓ ไม่เจอ
┌─────────────────────────┐
│  Stage 4: LLM (Ollama)  │ → ส่ง column_name + sample_values
└─────────────────────────┘
        ↓
   masking / partial_masking / pass
```

### Pros & Cons แต่ละ Stage

| Stage | Pros | Cons |
|---|---|---|
| Exact | เร็วที่สุด O(1), แม่นยำ 100% | ไม่รองรับ typo, alias ต้องครอบคลุมพอ |
| Fuzzy | รองรับ typo, ไม่ต้องโหลด model | อาจ false positive ถ้า threshold ต่ำเกิน |
| Semantic | เข้าใจความหมาย ข้ามภาษาได้ | ต้องโหลด model (~500MB), ช้ากว่า fuzzy |
| LLM | จับ edge case ได้ดี ใช้ sample values ประกอบ | ช้าที่สุด, ต้องรัน Ollama local, ผลไม่ consistent |

### Stage 2: Fuzzy — การตัดสิน Clear Winner

Fuzzy stage คำนวณ score ของทุก category แล้วตรวจว่า "winner ชนะขาดพอไหม" ก่อน return  
ถ้าไม่ชัดเจนพอ → ส่งต่อ Stage 3 (Semantic) แทน

```python
fuzzy_clear_winner = (
    runner_up_score < fuzzy_threshold   # อันดับ 2 ไม่ถึง threshold
    or abs(best_score - runner_up_score) > 5  # หรือห่างกันมากกว่า 5 คะแนน
)
```

**กรณีที่ 1 — Clear Winner ✅ (return ทันที)**

column: `"ที่อยู่อาศัย"`

| Category | Score |
|---|---|
| ADDRESS_FULL | **96** ← best |
| ADDRESS_SHORT | 71 ← runner-up |

```
gap = 96 - 71 = 25  →  > 5 ✅
fuzzy_clear_winner = True  →  return ADDRESS_FULL (partial_masking)
```

---

**กรณีที่ 2 — Ambiguous ❌ (ส่งต่อ Semantic/LLM)**

column: `"ที่อยู่บ้านเลขที่"` (มีทั้ง "ที่อยู่" และ "บ้านเลขที่" ในชื่อเดียว)

| Category | Score |
|---|---|
| ADDRESS_FULL | **94** ← best |
| ADDRESS_SHORT | **91** ← runner-up |

```
runner_up = 91  →  ≥ 92? ❌
gap = 94 - 91 = 3  →  > 5? ❌

fuzzy_clear_winner = False  →  ส่งต่อ Semantic แทน
```

---

**กรณีที่ 3 — Runner-up ผ่าน threshold แต่ gap แคบ ❌ (ส่งต่อ Semantic/LLM)**

column: `"full home address"`

| Category | Score |
|---|---|
| ADDRESS_FULL | **97** ← best |
| ADDRESS_SHORT | **93** ← runner-up |

```
runner_up = 93  →  ≥ 92 ✅ (ผ่าน threshold!)
gap = 97 - 93 = 4  →  > 5? ❌

fuzzy_clear_winner = False  →  ส่งต่อ Semantic แทน
```

---

**สรุป logic:**

```
fuzzy_clear_winner = True  เมื่อ...

  เงื่อนไข A: runner_up < 92     → อันดับ 2 ไม่ถึง threshold เลย
              (อันดับ 1 ชนะขาด ไม่มีคู่แข่ง)
  OR
  เงื่อนไข B: gap > 5 คะแนน     → ห่างกันพอที่จะมั่นใจได้
              (อันดับ 1 ชนะแบบห่างชัดเจน)
```

---

## แผนการ Implement

```
Phase 1: ชื่อ + อีเมล
  → alias ชัดเจน ทดสอบได้เร็ว

Phase 2: ที่อยู่สั้น
  → classify ตาม column name + ตัดสิน masking/pass ตามระดับ

Phase 3: ที่อยู่ยาว
  → ต้องการ address parser (เริ่มด้วย keyword boundary)

Phase 4: โซเชียลมีเดีย
  → พึ่ง Semantic + LLM เป็นหลัก ทดสอบ false positive rate
```

---

## การติดตั้ง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## การรัน

```bash
python main.py
```


## เพิ่มเติม 
- ควรจะมี sample data ที่ให้ LLM ไปดูตรงนี้แทน ป้องกันการ leak data ส่งไปให้ cloud ai โดย API
- ตรง category ของ prefix พวก PREFIX ,คำนำหน้าชื่อ
,คำนำหน้าคำนำหน้านามคำนำหน้าชื่อ, prefixtitletitle_id, name ,title → masking (****)