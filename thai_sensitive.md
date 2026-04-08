# Thai Sensitive Column Detector

ตรวจจับคอลัมน์ข้อมูลอ่อนไหวในบริบทข้อมูลไทย (PDPA)

## ผลลัพธ์

- `masking` — mask ทั้งค่า → `****`
- `partial_masking` — mask เฉพาะส่วนที่อยู่ระดับถนนลงไป เก็บตำบล/อำเภอ/จังหวัดไว้
- `pass` — ไม่ใช่ข้อมูลอ่อนไหว

## Sensitive Types

| Sensitive Type | Output | ตัวอย่าง alias |
|---|---|---|
| FULLNAME | masking | ชื่อ-นามสกุล, full name, PERSON_NAME |
| PREFIX | pass | คำนำหน้า, title, prefix |
| FIRSTNAME | masking | ชื่อ, fname, first_name, GiveName |
| LASTNAME | masking | นามสกุล, lname, last_name, FamilyName |
| EMAIL | masking | อีเมล, email, e_mail |
| ADDRESS_SHORT | masking | บ้านเลขที่, ซอย, ถนน, address no, moo |
| ADDRESS_FULL | partial_masking | ที่อยู่, address, home address |

## ตัวอย่าง Output

```
masking         → ****
partial_masking → **** ตำบลบางจาก เขตพระโขนง กรุงเทพ 10260
```

## การใช้งาน

```python
from thai_column_classifier import SensitiveColumnClassifier, SensitiveColumnInput

# ค่าเริ่มต้น: semantic = LocalSemanticProvider (paraphrase-multilingual-MiniLM-L12-v2)
#              llm     = OllamaProvider (llama3.2)
clf = SensitiveColumnClassifier()

result = clf.classify(SensitiveColumnInput(
    column_name="ชื่อ-นามสกุล",
    sample_values=["สมชาย ใจดี"],
))

print(result.decision)        # masking
print(result.sensitive_type)  # FULLNAME
print(result.confidence)      # 1.0
```

## Detection Pipeline

```
column_name + sample_values
        ↓
┌─────────────────────────┐
│  Stage 1: Exact Match   │ → ถ้าเจอ: return ทันที (confidence = 1.0)
└─────────────────────────┘
        ↓ ไม่เจอ
┌─────────────────────────┐
│  Stage 2: Fuzzy Match   │ → ถ้าเกิน threshold และชนะขาด: return
│  - ratio                │
│  - partial_ratio        │
│  - token_sort_ratio     │
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
| Exact | เร็วที่สุด, แม่นยำ 100% | ไม่รองรับ typo |
| Fuzzy | รองรับ typo, ไม่ต้องโหลด model | อาจ false positive ถ้า threshold ต่ำเกิน |
| Semantic | เข้าใจความหมาย ข้ามภาษาได้ | ต้องโหลด model (~500MB) |
| LLM | จับ edge case ได้ดี ใช้ sample values ประกอบ | ช้าที่สุด, ต้องรัน Ollama local |

### Fuzzy Match — เงื่อนไข partial_ratio

`partial_ratio` จะถูกใช้ก็ต่อเมื่อ term **ครอบคลุม >= 60%** ของความยาว column name เท่านั้น

**เหตุผล:** `partial_ratio` เอา term สั้นไปวิ่งหาใน string ยาว ถ้า term เป็นแค่ส่วนเล็กๆ ของชื่อคอลัมน์ จะเกิด false positive ได้ เช่น:

| column name | term | coverage | ใช้ partial_ratio? |
|---|---|---|---|
| `ลักษณะที่อยู่อาศัย` | `ที่อยู่` | 7/19 = 0.37 | ❌ ไม่ใช้ → ส่งต่อ LLM → `pass` |
| `ที่อยู่ปัจจุบัน` | `ที่อยู่` | 7/15 = 0.47 | ❌ ไม่ใช้ → ส่งต่อ LLM → `partial_masking` |
| `ที่อยู่เต็ม` | `ที่อยู่` | 7/11 = 0.64 | ✅ ใช้ → fuzzy hit → `partial_masking` |

> **กรณี false positive ที่พบ:** คอลัมน์ `ลักษณะที่อยู่อาศัย` (ค่าคือ เช่า / ที่ดินกรรมสิทธิ์ / อื่นๆ) ถูก classify เป็น `partial_masking` เพราะ `"ที่อยู่"` ปรากฏเป็น substring ให้ partial_ratio ยิงคะแนน ~100

## LLM Providers

```python
from thai_column_classifier import OllamaProvider, OpenAIProvider, ClaudeProvider, HFLLMProvider

# Ollama — fully local, recommended
OllamaProvider(model="llama3.2")

# OpenAI
OpenAIProvider(api_key="sk-...")

# Claude
ClaudeProvider(api_key="...")

# HuggingFace
HFLLMProvider(api_key="hf_...")
```

## Semantic Providers

```python
from thai_column_classifier import LocalSemanticProvider, HFSemanticProvider

# Local — no API key needed (~500MB download on first use)
LocalSemanticProvider()

# HuggingFace API — no local model needed
HFSemanticProvider(api_key="hf_...")
```
