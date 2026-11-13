# รายงานการเทรนโมเดล (Training Report)

วันที่: 2026-11-06 15:01:39


## 1-2. Data Collection & Labeling

| คลาส (label) | ภาพดิบ | ภาพจากกล้อง |
|---|---:|---:|
| abhaya | 1 | 0 |
| chin | 1 | 0 |
| dharmachakra | 1 | 0 |
| ksepana | 1 | 0 |
| mudra_R | 1 | 0 |
| namaskara_mudra | 1 | 0 |
| vitarka_mudra | 1 | 0 |

## 3-4. EDA & Data Cleaning / Preprocessing

- `abhaya.jpg` ขนาด 1280x1280 px, ROI [380, 190, 800, 820], ลบส่วนที่ไม่ใช่มือ 0 กรอบ
- `chin.jpg` ขนาด 1280x1280 px, ROI [300, 175, 1060, 540], ลบส่วนที่ไม่ใช่มือ 0 กรอบ
- `dharmachakra.jpg` ขนาด 1280x1280 px, ROI [140, 40, 1010, 900], ลบส่วนที่ไม่ใช่มือ 0 กรอบ
- `ksepana.jpg` ขนาด 1280x1280 px, ROI [330, 170, 960, 900], ลบส่วนที่ไม่ใช่มือ 0 กรอบ
- `mudra_R.jpg` ขนาด 1280x1280 px, ROI [380, 120, 830, 830], ลบส่วนที่ไม่ใช่มือ 3 กรอบ
- `namaskara_mudra.jpg` ขนาด 1280x1280 px, ROI [410, 100, 950, 770], ลบส่วนที่ไม่ใช่มือ 0 กรอบ
- `vitarka_mudra.jpg` ขนาด 1280x1280 px, ROI [400, 40, 820, 700], ลบส่วนที่ไม่ใช่มือ 5 กรอบ

ภาพผลการแบ่งส่วนมือ (ภาพ ROI | mask สีผิว | ภาพมือที่ crop | silhouette):

![segmentation](segmentation_preview.png)

## 5-6. Data Augmentation & Feature Engineering

- augment ภาพดิบภาพละ 400 ครั้ง, ภาพจากกล้องภาพละ 6 ครั้ง
- ได้ข้อมูลทั้งหมด **2807** ตัวอย่าง, เวกเตอร์ฟีเจอร์ **2035** มิติ (HOG 1764 + Silhouette 256 + Hu 7 + Shape 8)

## 7. Data Splitting (train / validation / test)

train = 1967 / val = 420 / test = 420

## 8. Feature Scaling & Dimensionality Reduction (PCA)

StandardScaler -> PCA: 2035 มิติ -> **120** มิติ (อธิบายความแปรปรวน 83.0%) ใช้เวลา 2.1 วินาที

## 9. Model Training

| โมเดล | Train acc | Val acc | Val macro-F1 | เวลาเทรน (วินาที) |
|---|---:|---:|---:|---:|
| kNN | 100.00% | 100.00% | 1.0000 | 0.0 |
| Softmax Regression | 100.00% | 99.52% | 0.9952 | 1.1 |
| MLP 120-128-64-7 | 100.00% | 100.00% | 1.0000 | 1.5 |

## 10. Model Selection

เลือกโมเดล **MLP 120-128-64-7** (validation macro-F1 สูงสุด = 1.0000)

Learning curve (น้ำเงิน = train loss, ส้ม = validation loss):

![learning curve](learning_curve.png)

## 11. Model Evaluation (Test set)

### Test set (ข้อมูลที่โมเดลไม่เคยเห็นตอนเทรน)

| คลาส | Precision | Recall | F1 | จำนวน |
|---|---:|---:|---:|---:|
| abhaya | 1.000 | 1.000 | 1.000 | 60 |
| chin | 1.000 | 1.000 | 1.000 | 60 |
| dharmachakra | 1.000 | 1.000 | 1.000 | 60 |
| ksepana | 1.000 | 1.000 | 1.000 | 60 |
| mudra_R | 0.984 | 1.000 | 0.992 | 60 |
| namaskara_mudra | 1.000 | 1.000 | 1.000 | 60 |
| vitarka_mudra | 1.000 | 0.983 | 0.992 | 60 |
| **macro avg** | 0.998 | 0.998 | 0.998 | 420 |

**Accuracy = 99.76%**

### Confusion matrix (test)

| จริง \ ทำนาย | abhaya | chin | dharmach | ksepana | mudra_R | namaskar | vitarka_ |
|---|---:|---:|---:|---:|---:|---:|---:|
| abhaya | 60 | 0 | 0 | 0 | 0 | 0 | 0 |
| chin | 0 | 60 | 0 | 0 | 0 | 0 | 0 |
| dharmachakra | 0 | 0 | 60 | 0 | 0 | 0 | 0 |
| ksepana | 0 | 0 | 0 | 60 | 0 | 0 | 0 |
| mudra_R | 0 | 0 | 0 | 0 | 60 | 0 | 0 |
| namaskara_mudra | 0 | 0 | 0 | 0 | 0 | 60 | 0 |
| vitarka_mudra | 0 | 0 | 0 | 0 | 1 | 0 | 59 |

### ตรวจสอบกับภาพจริงทั้งหมดที่ไม่ผ่าน augmentation (7/7 ถูก)

| ภาพ | ป้ายจริง | ทำนาย | ความมั่นใจ |
|---|---|---|---:|
| abhaya.jpg | abhaya | abhaya | 98.2% |
| chin.jpg | chin | chin | 97.6% |
| dharmachakra.jpg | dharmachakra | dharmachakra | 99.2% |
| ksepana.jpg | ksepana | ksepana | 98.6% |
| mudra_R.jpg | mudra_R | mudra_R | 99.1% |
| namaskara_mudra.jpg | namaskara_mudra | namaskara_mudra | 98.2% |
| vitarka_mudra.jpg | vitarka_mudra | vitarka_mudra | 98.4% |

### Robustness test (บิดภาพแรงกว่าตอนเทรน ด้วย seed ใหม่)

| ระดับการบิดภาพ | ความหมาย | Accuracy | ความมั่นใจเฉลี่ย |
|---|---|---:|---:|
| x1.0 | เท่าตอนเทรน | 99.5% | 96% |
| x1.5 | แรงกว่า 1.5 เท่า (หมุน ±27°) | 97.1% | 89% |
| x2.0 | แรงกว่า 2 เท่า (หมุน ±36°) | 87.1% | 74% |

> **ข้อควรระวัง:** Raw Data มีเพียงปางละ 1 ภาพ ชุด test ที่มาจากการ augment ภาพเดียวกัน จึงวัดได้แค่ความทนทานต่อการหมุน/ย่อ/แสง ไม่ใช่ความแม่นยำกับคนอื่นหรือฉากอื่น ควรเก็บข้อมูลจริงจากกล้องเพิ่ม (ปุ่ม "เก็บตัวอย่าง" ใน app.py) แล้วเทรนใหม่

## 12. Model Deployment (บันทึกโมเดล)

- ไฟล์โมเดล: `models/mudra_model.npz` (สำเนาเวอร์ชัน `models/history/mudra_model_20261106_150302.npz`)
- เวลาเทรนรวม 83.2 วินาที