"""ทดสอบว่าใบหน้าไม่ถูกนับเป็นมือ (รัน: python tests/test_face_removal.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mudra.segmentation import SkinModel, extract_hand

SKIN = (205, 150, 125)


def scene():
    img = np.full((160, 200, 3), 60, np.uint8)
    yy, xx = np.mgrid[:160, :200]
    img[(xx - 60) ** 2 + (yy - 35) ** 2 < 26 ** 2] = SKIN        # ใบหน้า
    img[60:90, 45:75] = SKIN                                       # คอ (ติดหน้า)
    img[90:140, 140:180] = SKIN                                    # มือ
    return img


def test_face_removed():
    img, face = scene(), [(34, 9, 86, 61)]
    sk = SkinModel()
    without = extract_hand(img, sk)
    with_face = extract_hand(img, sk, faces=face)
    assert without.found and with_face.found
    x1, y1, x2, y2 = with_face.bbox
    assert x1 >= 130 and y1 >= 80, f"ยังรวมใบหน้า/คอ: {with_face.bbox}"
    assert without.bbox[1] < 40, "ฉากทดสอบต้องมีใบหน้าเมื่อไม่ตัด"


def test_hand_near_face_kept():
    img, face = scene(), [(34, 9, 86, 61)]
    img[100:130, 100:125] = SKIN
    h = extract_hand(img, SkinModel(), faces=face)
    assert h.found and h.bbox[0] <= 105


if __name__ == "__main__":
    test_face_removed(); test_hand_near_face_kept(); print("ผ่านทั้งหมด")
