"""
train.py — เทรนโมเดลตรวจจับปางจากท่ามือ (รันวงจร ML ครบทุกขั้นตอน)

ตัวอย่าง:
    python train.py                 # เทรนเต็ม
    python train.py --quick         # เทรนเร็ว (ข้อมูลน้อย) สำหรับทดสอบ
    python train.py --no-collected  # ใช้เฉพาะ Raw Data
"""
import argparse
import sys

from mudra import config
from mudra.trainer import run_training


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="เทรนโมเดลตรวจจับปางพระพุทธรูปจากท่ามือ")
    ap.add_argument("--aug", type=int, default=config.AUG_PER_RAW_IMAGE, help="จำนวน augment ต่อภาพดิบ")
    ap.add_argument("--aug-collected", type=int, default=config.AUG_PER_COLLECTED,
                    help="จำนวน augment ต่อภาพที่เก็บจากกล้อง")
    ap.add_argument("--epochs", type=int, default=config.MLP_EPOCHS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-collected", action="store_true", help="ไม่ใช้ข้อมูลที่เก็บจากกล้อง")
    ap.add_argument("--quick", action="store_true", help="โหมดทดสอบเร็ว")
    a = ap.parse_args()
    if a.quick:
        a.aug, a.epochs = 60, 40
    run_training(a.aug, a.aug_collected, a.epochs, a.seed, not a.no_collected)


if __name__ == "__main__":
    main()
