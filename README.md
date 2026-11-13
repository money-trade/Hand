# ☸ ระบบแปลท่ามือเป็นปางพระพุทธรูป (Mudra → Text)

โปรแกรมภาษา Python ที่ใช้หลักการ **Machine Learning** และ **NLP** ตรวจจับ "ท่ามือ (มุทรา)" ของผู้ใช้ผ่านกล้อง
แล้วแปลงเป็น **ข้อความภาษาไทย** บอกว่าเป็นปางอะไร มีความหมายอย่างไร

- **หน้าจอซ้าย** — ภาพจากกล้อง จับการเคลื่อนไหว (สีแดง) และบริเวณมือ (สีเขียว) ภายในกรอบ ROI
- **หน้าจอขวา** — ชื่อปาง ความมั่นใจ ลักษณะท่ามือ ความหมาย ความน่าจะเป็นทุกปาง และข้อความที่ถอดได้ต่อเนื่อง (transcript)
- ข้อมูลตั้งต้นมาจากโฟลเดอร์ **`Raw Data`**
- **ไม่ใช้ไลบรารี ML/NLP/Computer Vision สำเร็จรูป** ทุกอัลกอริทึมเขียนเอง ครบทั้ง **วงจรชีวิตของ ML (ML Lifecycle)**

![หน้าจอโปรแกรม (โหมดสาธิต)](docs/app_demo.png)

---

## 1. ข้อกำหนด "ไม่ใช้ไลบรารี — ทำเองทั้งหมด"

| สิ่งที่ต้องทำ | ใช้อะไร | หมายเหตุ |
|---|---|---|
| แยกสีผิว, เบลอ, morphology, connected components, ย่อ/หมุนภาพ | **เขียนเอง** (`mudra/image_ops.py`, `segmentation.py`) | ไม่เรียกฟังก์ชันประมวลผลภาพของ OpenCV เลย |
| สกัดฟีเจอร์ HOG, Hu moments, รูปทรง | **เขียนเอง** (`mudra/features.py`) | |
| Data augmentation, แบ่ง train/val/test | **เขียนเอง** (`mudra/dataset.py`) | |
| StandardScaler, PCA (+ หาค่าลักษณะเฉพาะด้วย Jacobi) | **เขียนเอง** (`mudra/preprocessing.py`) | ไม่ใช้ `np.linalg` |
| kNN, Softmax Regression, MLP + Backprop + Adam | **เขียนเอง** (`mudra/models.py`) | ไม่ใช้ scikit-learn / TensorFlow / PyTorch |
| Accuracy, Precision, Recall, F1, Confusion matrix | **เขียนเอง** (`mudra/metrics.py`) | |
| HMM (Forward / Viterbi), Tokenization, สร้างประโยคภาษาไทย | **เขียนเอง** (`mudra/nlp.py`) | ไม่ใช้ NLTK / PyThaiNLP |
| ตรวจจับการเคลื่อนไหว (background subtraction) | **เขียนเอง** (`app.py`) | |
| หน้าจอโปรแกรม | `tkinter` (มากับ Python) | แปลงภาพเป็น PPM เอง ไม่ใช้ PIL |
| อาร์เรย์ตัวเลข | `numpy` | ใช้แค่ บวก/ลบ/คูณเมทริกซ์ |
| **เปิดกล้อง + อ่าน/เขียนไฟล์รูป JPG/PNG** | `opencv-python` | **ใช้เฉพาะ I/O** ใน `mudra/io_utils.py` ไฟล์เดียว |

> ไม่ใช้ MediaPipe, scikit-learn, TensorFlow, Keras, PyTorch, PIL, scikit-image, NLTK, PyThaiNLP

---

## 2. ปางที่รองรับ (7 คลาส จาก `Raw Data`)

| ไฟล์ใน Raw Data | คลาส | ชื่อภาษาไทย | ลักษณะท่ามือ |
|---|---|---|---|
| `abhaya.jpg` | abhaya | ปางประทานอภัย (ปางห้ามญาติ) | ยกมือ ตั้งฝ่ามือ นิ้วชิดชี้ขึ้น |
| `chin.jpg` | chin | จินมุทรา | นิ้วโป้งจรดนิ้วชี้เป็นวง คว่ำมือแนวนอน |
| `dharmachakra.jpg` | dharmachakra | ปางปฐมเทศนา (ธรรมจักรมุทรา) | สองมือจีบวงนิ้ว นำวงมาจรดกัน |
| `ksepana.jpg` | ksepana | กเษปณมุทรา | ประสานมือ เหยียดนิ้วชี้ทั้งสองชิดกัน |
| `mudra_R.jpg` | mudra_R | มุทรา R (ชูนิ้วชี้) | กำมือ ชูนิ้วชี้ขึ้น |
| `namaskara_mudra.jpg` | namaskara_mudra | นมัสการมุทรา (พนมมือ) | ประนมมือระดับอก |
| `vitarka_mudra.jpg` | vitarka_mudra | ปางแสดงธรรม (วิตรรกมุทรา) | นิ้วโป้งจรดนิ้วชี้เป็นวง นิ้วอื่นชี้ขึ้น |

ชื่อ ความหมาย และคำอธิบายทั้งหมดแก้ไขได้ที่ `data/mudra_info.json` (คลังความรู้ที่ส่วน NLP ใช้สร้างประโยค)

---

## 3. การติดตั้งและใช้งาน

```bash
pip install -r requirements.txt
```

**ขั้นที่ 1 — เทรนโมเดล** (รันวงจร ML ครบทุกขั้น ใช้เวลาประมาณ 1–2 นาที)

```bash
python train.py
```

**ขั้นที่ 2 — เปิดโปรแกรมกล้อง**

```bash
python app.py
```

ไม่มีกล้อง / อยากทดสอบก่อน: โหมดสาธิตจะวนภาพจาก `Raw Data` แทนกล้อง

```bash
python app.py --demo
```

ทำนายจากไฟล์รูป (ไม่ต้องเปิดหน้าจอ)

```bash
python predict_image.py "Raw Data/vitarka_mudra.jpg"
```

### การใช้งานหน้าจอ

1. ยืนห่างกล้องให้เห็นช่วงอก ยกมือทำท่าให้อยู่ใน **กรอบ ROI** (ลากเมาส์บนภาพเพื่อย้าย/ปรับขนาดกรอบ ไม่ให้หน้าอยู่ในกรอบ)
2. สีกรอบบอกสถานะ: 🟥 ไม่พบมือ 🟨 กำลังเคลื่อนไหว 🟩 นิ่งและกำลังตรวจจับ
3. เมื่อค้างท่า **นิ่ง** ประมาณครึ่งวินาที ชื่อปางจะถูกเพิ่มลงช่อง **ข้อความที่ถอดได้**
4. เอามือออกจากกรอบ = คั่นคำ (ทำท่าเดิมซ้ำได้)

| ปุ่ม / คีย์ | ทำอะไร |
|---|---|
| `เก็บ 1 ภาพ`, `เก็บต่อเนื่อง 20 ภาพ`, คีย์ `1`–`7` | เก็บภาพในกรอบ ROI เป็นข้อมูลจริงของปางที่เลือก → `data/collected/<ปาง>/` |
| `ฝึกโมเดลใหม่` | เทรนใหม่ (Raw Data + ข้อมูลที่เก็บ) แล้วโหลดโมเดลใหม่ทันที |
| `ปรับเทียบสีผิว` | วางฝ่ามือเต็มกรอบเส้นประ 3 วินาที → เรียนรู้สีผิวของผู้ใช้ (Gaussian) |
| `บันทึกข้อความ` / คีย์ `S` | บันทึก transcript ลง `transcripts/` (รวมผลถอดรหัสแบบ Viterbi) |
| `ล้างข้อความ` / คีย์ `C`, ออก `Q`/`Esc` | |

---

## 4. Flowchart

### 4.1 วงจรชีวิตของ Machine Learning (ML Lifecycle) ในโปรเจคนี้

```mermaid
flowchart TD
    A([เริ่ม]) --> B["1. Data Collection<br/>Raw Data/*.jpg<br/>+ data/collected/ (จากกล้อง)"]
    B --> C["2. Data Labeling<br/>annotations.json: label + กรอบ ROI<br/>mudra_info.json: ชื่อ/ความหมาย"]
    C --> D["3. EDA<br/>ขนาดภาพ จำนวนต่อคลาส<br/>ภาพตัวอย่างการแบ่งส่วนมือ"]
    D --> E["4. Data Cleaning & Preprocessing<br/>ลบคอ/คาง → ตัด ROI → ย่อภาพ<br/>YCrCb skin mask → morphology<br/>→ connected components → crop 64×64"]
    E --> F["5. Data Augmentation<br/>หมุน ±18° ย่อขยาย พลิกซ้ายขวา<br/>เลื่อน แสง สี gamma noise blur<br/>(ภาพละ 400 แบบ)"]
    F --> G["6. Feature Engineering<br/>HOG 1764 + Silhouette 256<br/>+ Hu moments 7 + Shape 8 = 2035 มิติ"]
    G --> H["7. Data Splitting<br/>train 70% / val 15% / test 15%<br/>(stratified, ภาพจากกล้องแบ่งตามภาพต้นทาง)"]
    H --> I["8. Scaling + PCA<br/>StandardScaler → PCA (Orthogonal iteration + Jacobi)<br/>fit กับ train เท่านั้น"]
    I --> J["9. Model Training<br/>kNN • Softmax Regression • MLP<br/>(Backprop + Adam + Dropout + L2 + Early stopping)"]
    J --> K{"10. Model Selection<br/>validation macro-F1<br/>สูงสุด?"}
    K --> L["11. Evaluation<br/>Accuracy / Precision / Recall / F1<br/>Confusion matrix → reports/"]
    L --> M["12. Deployment<br/>models/mudra_model.npz<br/>+ เก็บเวอร์ชันใน models/history/"]
    M --> N["13. ใช้งานจริง app.py<br/>กล้อง → ปาง → ข้อความ"]
    N --> O{"Monitoring<br/>ทายผิด / ไม่มั่นใจ?"}
    O -->|ใช่| P["เก็บภาพจริงจากกล้อง<br/>ปุ่ม 'เก็บตัวอย่าง'"]
    P --> B
    O -->|ไม่| N
```

### 4.2 การทำงานแบบเวลาจริง (กล้อง → ข้อความ)

```mermaid
flowchart LR
    CAM["📷 เฟรมกล้อง<br/>640×480"] --> MIR["กลับด้านแบบกระจก"]
    MIR --> MOT["จับการเคลื่อนไหว<br/>Background subtraction<br/>ผลต่างกับพื้นหลัง > T"]
    MIR --> ROI["ตัดกรอบ ROI<br/>→ ภาพทำงาน 128 px"]
    ROI --> SKIN["แยกสีผิว YCrCb<br/>→ morphology<br/>→ connected components"]
    SKIN --> FOUND{"พบมือ?"}
    FOUND -->|ไม่| NONE["สถานะ: ไม่มีมือ"]
    FOUND -->|ใช่| FEAT["สกัดฟีเจอร์<br/>HOG + Silhouette<br/>+ Hu + Shape"]
    FEAT --> PRE["StandardScaler → PCA"]
    PRE --> CLS["โมเดล ML<br/>P(ปาง | ภาพ)"]
    CLS --> HMM["NLP: HMM Forward filter<br/>กรองผลตามเวลา"]
    NONE --> HMM
    MOT --> STILL{"นิ่ง?"}
    HMM --> STILL
    STILL -->|"นิ่ง + มั่นใจ ≥ 55%<br/>ต่อเนื่อง 12 เฟรม"| TOK["Tokenizer (แบบ CTC)<br/>ยุบท่าซ้ำ / ตัดช่วงไม่มีมือ"]
    STILL -->|เคลื่อนไหว| NLG
    TOK --> NLG["NLG ภาษาไทย<br/>เลือกถ้อยคำตามความมั่นใจ<br/>+ ความหมายจากคลังความรู้"]
    NLG --> UI["🖥️ หน้าจอขวา<br/>ชื่อปาง • ความหมาย<br/>• transcript"]
    MOT --> UIL["🖥️ หน้าจอซ้าย<br/>ภาพกล้อง + การเคลื่อนไหว<br/>+ มือที่ตรวจพบ"]
    SKIN --> UIL
```

### 4.3 ภายในขั้นการเทรน (train.py)

```mermaid
flowchart TD
    R["Raw Data (ปางละ 1 ภาพ)<br/>+ collected"] --> AUG["Augment ×400 ต่อภาพ"]
    AUG --> FE["Feature 2035 มิติ"]
    FE --> SP["Split train / val / test"]
    SP --> SC["StandardScaler.fit(train)"]
    SC --> PCA["PCA.fit(train) → ~120 มิติ"]
    PCA --> M1["kNN (k=7, cosine)"]
    PCA --> M2["Softmax Regression"]
    PCA --> M3["MLP 120-128-64-7"]
    M1 --> SEL{"เลือกโมเดลที่<br/>val macro-F1 สูงสุด"}
    M2 --> SEL
    M3 --> SEL
    SEL --> TE["ประเมิน test set<br/>+ ภาพจริงที่ไม่ augment"]
    TE --> SAVE["บันทึก .npz + รายงาน .md"]
```

---

## 5. รายละเอียดแต่ละขั้นของ ML Lifecycle

### ขั้นที่ 1–2 Data Collection & Labeling
- `Raw Data/` มีภาพต้นแบบ **ปางละ 1 ภาพ** (1280×1280) ชื่อไฟล์ = ชื่อคลาส
- `data/annotations.json` เก็บ **label** และ **กรอบ ROI** (บริเวณมือ ไม่รวมหน้า) ของแต่ละภาพ
  และกรอบ `exclude` สำหรับลบผิวที่ไม่ใช่มือ (คอ) ที่อยู่ติดกับปลายนิ้ว
- ข้อมูลเพิ่มจากกล้องจะถูกเก็บที่ `data/collected/<คลาส>/` (เพิ่มรูปเองในโฟลเดอร์ `Raw Data/<คลาส>/` ก็ได้)

### ขั้นที่ 3–4 EDA, Data Cleaning & Preprocessing
1. ทาสีเทาทับกรอบ `exclude` (data cleaning)
2. ย่อภาพด้วยการเฉลี่ยพิกเซล (area downscale) → ตัด ROI ให้ด้านยาว = 128 px ด้วย bilinear interpolation
3. **Skin segmentation**: แปลง RGB → YCrCb แล้วเลือกพิกเซลที่ `135 ≤ Cr ≤ 180` และ `80 ≤ Cb ≤ 135`
   (หรือใช้ Gaussian ที่เรียนรู้จากผิวผู้ใช้ — ระยะ Mahalanobis ≤ 3)
4. **Morphology** opening/closing ลบจุดรบกวน, **Connected components** (label propagation + pointer jumping) เก็บกลุ่มใหญ่สุด ≤ 3 กลุ่ม
5. crop สี่เหลี่ยมจัตุรัสรอบมือ → 64×64, ปรับความสว่างภายในมือเป็น 0..1
6. ผลลัพธ์บันทึกที่ `reports/segmentation_preview.png`

### ขั้นที่ 5 Data Augmentation
เนื่องจากมีข้อมูลปางละ 1 ภาพ จึงสร้างข้อมูลเพิ่ม 400 แบบต่อภาพ ด้วยการสุ่ม:
หมุน ±18°, ย่อ/ขยาย ±12%, พลิกซ้าย-ขวา (มือซ้าย/ขวา, กล้องกระจก), เลื่อน, ขยับกรอบ ROI ±8%,
ความสว่าง 0.65–1.3, contrast, สีแต่ละช่อง ±10%, gamma, Gaussian noise, blur
(การหมุนใช้ affine warp ครั้งเดียวจากภาพเต็ม จึงไม่เกิดขอบดำ)

### ขั้นที่ 6 Feature Engineering (2035 มิติ)
| ฟีเจอร์ | มิติ | จับลักษณะอะไร |
|---|---:|---|
| HOG (cell 8×8, 9 bins, block 2×2, L2-Hys) | 1764 | ทิศทางขอบนิ้ว รายละเอียดภายในมือ |
| Silhouette 16×16 | 256 | รูปทรงมือโดยรวม |
| Hu moments (log) | 7 | รูปทรงที่ไม่แปรตามการหมุน/ย่อขยาย |
| Shape: aspect, solidity, จำนวนมือ, จำนวน/ขนาด "รู" (วงนิ้ว), ทิศแกนหลัก, ความรี | 8 | เช่น วงนิ้วโป้ง-นิ้วชี้ของปางแสดงธรรม |

### ขั้นที่ 7–8 Splitting, Scaling, PCA
- แบ่ง train/val/test 70/15/15 แบบ stratified; ภาพจากกล้องแบ่ง **ตามภาพต้นทาง** กันข้อมูลรั่ว
- StandardScaler: `z = (x − μ) / σ`
- PCA: covariance → **Orthogonal iteration** (QR ด้วย Gram-Schmidt) → **Rayleigh–Ritz + Jacobi eigenvalue** → เก็บแกนที่อธิบายความแปรปรวนได้ ≥ 97% (สูงสุด 120 แกน)

### ขั้นที่ 9–10 Model Training & Selection
| โมเดล | รายละเอียด |
|---|---|
| kNN | k = 7, cosine similarity, ถ่วงน้ำหนักตามความใกล้ |
| Softmax Regression | `P = softmax(XW + b)`, cross-entropy, Adam |
| MLP | 120 → 128 → 64 → 7, ReLU, Dropout 0.25, L2 1e-4, label smoothing, Adam, early stopping |

เลือกโมเดลที่ **macro-F1 บน validation** สูงสุด

### ขั้นที่ 11–12 Evaluation & Deployment
- คำนวณ Accuracy, Precision, Recall, F1 รายคลาส + Confusion matrix → `reports/training_report.md`
- บันทึก scaler + PCA + น้ำหนักโมเดล ลงไฟล์เดียว `models/mudra_model.npz` และเก็บสำเนาทุกเวอร์ชันใน `models/history/`

### ขั้นที่ 13 Monitoring & Feedback loop
ในโปรแกรมกล้อง ถ้าทายผิด ให้เก็บภาพจริงของปางนั้น (ปุ่ม `เก็บต่อเนื่อง 20 ภาพ`) แล้วกด `ฝึกโมเดลใหม่`
→ ข้อมูลจริงถูกรวมเข้า ML lifecycle รอบใหม่ทันที และรายงานจะแสดงความแม่นยำบน "ภาพจริงจากกล้อง" แยกต่างหาก

---

## 6. ส่วน NLP (`mudra/nlp.py`)

โปรแกรมมองปัญหา "ท่ามือ → ข้อความ" เหมือน **การรู้จำเสียงพูด (speech-to-text)**: ผลทำนายรายเฟรมมีสัญญาณรบกวน
ต้องถอดเป็นลำดับคำที่ถูกต้องก่อนแล้วจึงสร้างประโยค

1. **Hidden Markov Model** — สถานะซ่อน = ปางที่ทำจริง (+ "ไม่มีมือ"), เมทริกซ์ transition แบบ sticky (อยู่สถานะเดิม 0.92)
   - **Forward algorithm** กรองผลแบบเวลาจริง: `αₜ ∝ P(obsₜ | s) · Σ αₜ₋₁ A` → ตัวอักษรไม่กระพริบ
   - **Viterbi algorithm** ถอดรหัสลำดับที่น่าจะเป็นที่สุดของทั้งช่วงตอนบันทึกไฟล์
2. **Tokenization แบบ CTC** — ยุบโทเคนที่ซ้ำติดกัน ตัดช่วง "ไม่มีมือ" (blank): `[ก ก ก _ _ ข ข ก] → [ก ข ก]`
3. **Natural Language Generation** แบบแม่แบบ (template-based)
   - เลือกถ้อยคำตามความมั่นใจ: ≥ 85% "ท่ามือนี้คือ…", ≥ 55% "น่าจะเป็น…", ต่ำกว่า "ยังไม่แน่ใจ อาจเป็น … หรือ …"
   - ดึงชื่อ ลักษณะท่า และความหมายจากคลังความรู้ `mudra_info.json`
   - สรุปลำดับ: "ตรวจพบทั้งหมด 3 ท่า (2 ปางที่ไม่ซ้ำกัน) ตามลำดับ: ประทานอภัย → แสดงธรรม → ประทานอภัย"

---

## 7. ผลการทดลอง

รายงานฉบับเต็มสร้างอัตโนมัติทุกครั้งที่เทรน: [`reports/training_report.md`](reports/training_report.md)

**ข้อมูล:** 7 ภาพดิบ × (1 + 400 augment) = 2,807 ตัวอย่าง → train 1,967 / val 420 / test 420
**ฟีเจอร์:** 2,035 มิติ → PCA เหลือ 120 มิติ • **เวลาเทรนรวม** ~75 วินาที (CPU)

| โมเดล | Train acc | Val acc | Val macro-F1 |
|---|---:|---:|---:|
| kNN (k=7) | 100.00% | 100.00% | 1.0000 |
| Softmax Regression | 100.00% | 99.52% | 0.9952 |
| **MLP 120-128-64-7 (ถูกเลือก)** | 100.00% | 100.00% | 1.0000 |

| การทดสอบ | ผล |
|---|---|
| Test set (420 ตัวอย่าง) | **Accuracy 99.76%**, macro-F1 0.998 (ผิด 1 ภาพ: vitarka → mudra_R) |
| ภาพต้นฉบับทั้ง 7 ภาพ | ถูก 7/7 ความมั่นใจ 97.6–99.2% |
| Robustness ×1.0 (บิดภาพเท่าตอนเทรน, seed ใหม่) | 99.5% |
| Robustness ×1.5 (หมุน ±27°, ย่อขยาย ±18%) | 97.1% |
| Robustness ×2.0 (หมุน ±36°, ย่อขยาย ±24%) | 87.1% |

![segmentation preview](reports/segmentation_preview.png)

> ตัวเลขเหล่านี้วัดจากภาพของคนเดียวกันในฉากเดียวกัน (ดูหัวข้อ 8) — ความแม่นยำจริงกับกล้องของผู้ใช้จะต่ำกว่านี้
> จนกว่าจะเก็บข้อมูลจริงเพิ่มผ่านโปรแกรม แล้วเทรนใหม่ (รายงานจะแสดงความแม่นยำบนภาพจากกล้องแยกให้)

---

## 8. ข้อจำกัด และวิธีทำให้แม่นยำขึ้น

- **Raw Data มีปางละ 1 ภาพ (คนเดียว เสื้อเดียว ฉากเดียว)** — ความแม่นยำบน test set มาจากภาพที่ augment จากภาพเดียวกัน
  จึงสูงเกินจริงเมื่อเทียบกับการใช้งานกับคนอื่น/ห้องอื่น **วิธีแก้ที่ได้ผลที่สุด:** เปิด `app.py` แล้วเก็บภาพจริงปางละ 20–40 ภาพ
  (ขยับมือ เปลี่ยนมุมเล็กน้อยระหว่างเก็บ) แล้วกด `ฝึกโมเดลใหม่`
- การแยกมือใช้ **สีผิว** — ถ้าเสื้อหรือฉากหลังมีสีคล้ายผิว (เบจ ส้ม ชมพู ไม้) จะรบกวน ให้ใส่เสื้อสีเข้ม/เทา ใช้ฉากหลังเรียบ และกด `ปรับเทียบสีผิว`
- อย่าให้ **ใบหน้า** อยู่ในกรอบ ROI (ใบหน้าเป็นสีผิวเหมือนกัน) — ลากเมาส์ย้ายกรอบให้อยู่ระดับอก
- ถ้าแสงน้อยมาก กล้องจะมี noise สูง ทำให้ตรวจการเคลื่อนไหวผิด — เพิ่มแสงด้านหน้า

## 9. แก้ปัญหาเบื้องต้น

| อาการ | วิธีแก้ |
|---|---|
| `เปิดกล้องไม่ได้` | ปิดโปรแกรมอื่นที่ใช้กล้อง หรือลอง `python app.py --camera 1` (โปรแกรมจะสลับเป็นโหมดสาธิตอัตโนมัติ) |
| กรอบเป็นสีแดงตลอด (ไม่พบมือ) | กด `ปรับเทียบสีผิว` / เพิ่มแสง / ขยายกรอบ ROI |
| สีเขียวติดเสื้อหรือฉากหลัง | เปลี่ยนเสื้อ/ฉากหลัง หรือ `ปรับเทียบสีผิว` |
| ทายปางผิดบ่อย | เก็บข้อมูลจริงของปางนั้นเพิ่ม แล้ว `ฝึกโมเดลใหม่` |
| `ยังไม่มีโมเดล` | รัน `python train.py` ก่อน |

## 10. โครงสร้างโปรเจค

```
โปรเจค/
├── Raw Data/                  ภาพต้นแบบของแต่ละปาง (ข้อมูลดิบ)
├── data/
│   ├── annotations.json       label + ROI + กรอบที่ต้องลบ
│   ├── mudra_info.json        คลังความรู้ภาษาไทย (NLP)
│   └── collected/             ภาพจริงที่เก็บจากกล้อง (สร้างอัตโนมัติ)
├── mudra/                     โค้ดหลัก (เขียนเองทั้งหมด)
│   ├── config.py              พาธ + ไฮเปอร์พารามิเตอร์
│   ├── io_utils.py            I/O: กล้อง + อ่าน/เขียนไฟล์รูป (ที่เดียวที่ใช้ OpenCV)
│   ├── image_ops.py           ประมวลผลภาพพื้นฐาน
│   ├── segmentation.py        แยกสีผิว + หามือ
│   ├── features.py            HOG, Hu moments, shape
│   ├── dataset.py             โหลดข้อมูล, augmentation, split
│   ├── preprocessing.py       StandardScaler, PCA, Jacobi
│   ├── models.py              kNN, Softmax Regression, MLP
│   ├── metrics.py             accuracy, precision, recall, F1, confusion matrix
│   ├── nlp.py                 HMM, Viterbi, tokenizer, NLG ภาษาไทย
│   ├── pipeline.py            รวมขั้นตอนตอนใช้งานจริง + บันทึก/โหลดโมเดล
│   └── trainer.py             ML lifecycle ขั้น 1–12
├── train.py                   เทรนโมเดล
├── app.py                     โปรแกรมกล้อง (ซ้าย: กล้อง, ขวา: ข้อความ)
├── predict_image.py           ทำนายจากไฟล์รูป
├── models/                    โมเดลที่เทรนแล้ว + ประวัติเวอร์ชัน
├── reports/                   รายงานการเทรน + ภาพประกอบ
└── transcripts/               ข้อความที่บันทึกจากโปรแกรม
```
