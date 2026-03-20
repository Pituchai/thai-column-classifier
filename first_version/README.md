# Column Classification POC

โปรเจกต์นี้เป็นตัวอย่างสำหรับใช้จำแนกชื่อคอลัมน์ว่าอาจเป็นข้อมูล `Thai citizen ID` หรือไม่ และส่งผลลัพธ์ออกมาเป็น 3 แบบ:

- `auto_hash`
- `human_review`
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
- `human_review`: ชื่อคอลัมน์กำกวม หรือยังไม่มั่นใจพอ ควรให้คนตรวจสอบ
- `pass`: ไม่เข้าข่ายคอลัมน์เลขบัตรประชาชน

## หลักการตัดสินใจโดยสรุป

ลำดับการตรวจหลักใน `classify()` เป็นแบบนี้:

1. normalize ชื่อคอลัมน์
2. ถ้าอยู่ใน `_NON_CID_TERMS` ให้ `pass` ทันที
3. ถ้าอยู่ใน `_GENERIC_TERMS` ให้ `human_review` ทันที
4. exact match กับ `_CID_TERMS`
5. fuzzy match กับ `_CID_TERMS`
6. semantic matching ถ้า backend พร้อมใช้งาน
7. ใช้ guardrail จาก sample values เพื่อช่วยประกอบการตัดสินใจ

แนวคิดของแต่ละกลุ่ม:

- `_CID_TERMS` คือชื่อที่บ่งชี้ชัดว่าเป็นคอลัมน์ CID
- `_GENERIC_TERMS` คือชื่อที่กว้างหรือกำกวม เช่น `id`, `number`, `เลขที่บัตร`
- `_NON_CID_TERMS` คือชื่อที่ชัดว่าไม่ใช่ CID เช่น `phone_number`
- `_REPLACEMENTS` ใช้แปลงคำสะกดติดกัน คำย่อ หรือ typo ให้เป็นรูปแบบมาตรฐานก่อน match

## ตัวอย่างการ normalize

ตัวอย่างเช่น:

- `citizenid` -> `citizen id`
- `nationalid` -> `national id`
- `idcard` -> `id card`
- `เลขบตรประชาชน` -> `เลขบัตรประชาชน`
- `เลขบัตรปชช` -> `เลขบัตรประชาชน`

จุดประสงค์คือทำให้ชื่อที่พิมพ์ต่างรูปแบบกันสามารถ match กับ rule เดียวกันได้

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
- ชื่อที่กำกวมมาก เช่น `ID`, `id`, `number`, `เลขที่บัตร` จะถูกส่งไป `human_review` ก่อน เพื่อกัน false positive
- ชื่อทั่วไปอย่าง `code` หรือ `record id` จะไม่ถูกบังคับไป `human_review` ถ้าไม่มีสัญญาณอื่นว่าเกี่ยวกับ CID
- ชื่อที่ชัดว่าไม่ใช่ CID เช่น `phone_number` จะถูกส่งไป `pass` ทันที
- ถ้า semantic backend ยังไม่พร้อม ระบบจะ fallback ไปใช้ lexical logic เป็นหลัก

## คำถาม 
- ควรจะ test_data แบบไหน 
- สิ่งที่ทำแบบนี้ เป็น lib ที่ต้องการไหม 
- 
