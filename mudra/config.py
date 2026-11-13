"""
config.py — ค่าคงที่ / พาธ / ไฮเปอร์พารามิเตอร์ ของทั้งโปรเจค
"""
from pathlib import Path

# ---------------------------------------------------------------- พาธ
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Raw Data"                  # ข้อมูลดิบ (รูปต้นแบบของแต่ละปาง)
DATA_DIR = ROOT / "data"
ANNOTATION_FILE = DATA_DIR / "annotations.json"   # ป้ายกำกับ + กรอบมือ (ROI) ของรูปดิบ
INFO_FILE = DATA_DIR / "mudra_info.json"          # คลังความรู้ภาษาไทย (ใช้ในส่วน NLP)
COLLECTED_DIR = DATA_DIR / "collected"             # ตัวอย่างที่เก็บเพิ่มจากกล้อง
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "mudra_model.npz"
MODEL_META_FILE = MODEL_DIR / "mudra_model.json"
MODEL_HISTORY_DIR = MODEL_DIR / "history"
SKIN_FILE = MODEL_DIR / "skin_model.json"
REPORT_DIR = ROOT / "reports"
TRANSCRIPT_DIR = ROOT / "transcripts"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------- ภาพ / การแบ่งส่วนมือ
WORK_SIZE = 128        # ด้านยาวของภาพ ROI หลังปรับขนาด (ภาพทำงาน)
CROP_SIZE = 64         # ขนาดภาพมือที่ crop แล้ว (สี่เหลี่ยมจัตุรัส)
MIN_SKIN_RATIO = 0.02  # สัดส่วนพิกเซลผิวขั้นต่ำใน ROI จึงจะถือว่า "มีมือ"

# ค่าเริ่มต้นของโมเดลสีผิว (ช่วงสี Cr/Cb ในปริภูมิสี YCrCb)
SKIN_CR = (135.0, 180.0)
SKIN_CB = (80.0, 135.0)
SKIN_Y_MIN = 45.0

# ---------------------------------------------------------------- ฟีเจอร์
HOG_CELL = 8
HOG_BINS = 9
HOG_BLOCK = 2
MASK_GRID = 16

# ---------------------------------------------------------------- การเพิ่มข้อมูล (augmentation)
AUG_PER_RAW_IMAGE = 400      # ภาพดิบ 1 ภาพ -> สร้างตัวอย่างใหม่กี่ภาพ
AUG_PER_COLLECTED = 6        # ภาพที่เก็บจากกล้อง 1 ภาพ -> กี่ภาพ

# ---------------------------------------------------------------- การแบ่งข้อมูล / โมเดล
SPLIT = (0.70, 0.15, 0.15)   # train / validation / test
PCA_MAX_COMPONENTS = 120
PCA_VARIANCE = 0.97
SEED = 42

MLP_HIDDEN = (128, 64)
MLP_EPOCHS = 150
MLP_LR = 1e-3
MLP_L2 = 1e-4
MLP_DROPOUT = 0.25
MLP_BATCH = 64
MLP_PATIENCE = 20
KNN_K = 7

# ---------------------------------------------------------------- Open-set: ปฏิเสธท่าที่ไม่อยู่ในฐานข้อมูล
NOVELTY_K = 5                # ใช้ระยะถึงเพื่อนบ้านลำดับที่ k
NOVELTY_MAX_REFS = 2500      # จำนวนตัวอย่างอ้างอิงสูงสุดที่เก็บในโมเดล
NOVELTY_PERCENTILE = 90.0    # เกณฑ์ = เปอร์เซ็นไทล์ของคะแนนภาพที่รู้จัก (รวมภาพบิดแรง)
NOVELTY_MARGIN = 1.0         # x margin (ปรับเพิ่มได้ระหว่างใช้งานด้วยแถบเลื่อนในโปรแกรม)
NOVELTY_CAL_STRENGTH = 1.5   # ภาพบิดแรงขึ้นที่ใช้ร่วมตั้งเกณฑ์ (ให้เกณฑ์ไม่เข้มเกินไป)
MAX_SKIN_RATIO = 0.92        # สีผิวเต็มกรอบเกินนี้ = ใกล้กล้องเกินไป / ใบหน้าในกรอบ
CUT_WRIST = True             # ตัดแขนส่วนที่เกินข้อมือ (เมื่อแขนพาดมาจากขอบกรอบ)
MOTION_DIFF = 18             # ผลต่างความสว่างระหว่างเฟรม (ภาพทำงาน) ที่นับว่าเคลื่อนไหว
FACE_EVERY = 3               # ตรวจหาใบหน้าทุกกี่เฟรม (ประหยัดเวลา)
FACE_HOLD = 15               # จำตำแหน่งใบหน้าล่าสุดไว้กี่เฟรมเมื่อตรวจไม่เจอ
MIN_HAND_SIDE = 12           # ด้านของมือ (พิกเซลบนภาพทำงาน) ต้องไม่น้อยกว่านี้

# ---------------------------------------------------------------- เวลาจริง (กล้อง) + NLP
CONF_THRESHOLD = 0.55        # ความมั่นใจขั้นต่ำที่จะยอมรับว่าเป็นปางนั้น
HMM_STAY_PROB = 0.92         # ความน่าจะเป็นที่ท่าเดิมจะคงอยู่เฟรมถัดไป
STABLE_FRAMES = 12           # ต้องนิ่ง + ทายตรงกันกี่เฟรม จึงบันทึกเป็นข้อความ
MOTION_PIXEL_THRESHOLD = 22  # ความต่างของความสว่างที่ถือว่าพิกเซลนั้น "เคลื่อนไหว"
MOTION_MOVING_RATIO = 0.06   # สัดส่วนพิกเซลเคลื่อนไหวใน ROI ที่ถือว่า "กำลังขยับ"
