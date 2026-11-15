"""
ทดสอบการแยก "มือ" ออกจากใบหน้า/แขน (ใช้ภาพสังเคราะห์ ไม่ต้องใช้กล้อง)

รันทั้งหมด:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mudra.segmentation import SkinModel, extract_hand

SKIN = (205, 150, 125)


def blank(h=160, w=200):
    return np.full((h, w, 3), 60, np.uint8)


def face_scene():
    """ใบหน้า + คอ (ติดกัน) ด้านซ้ายบน และมือด้านขวาล่าง"""
    img = blank()
    yy, xx = np.mgrid[:160, :200]
    img[(xx - 60) ** 2 + (yy - 35) ** 2 < 26 ** 2] = SKIN
    img[60:90, 45:75] = SKIN
    img[90:140, 140:180] = SKIN
    return img


FACE = [(34, 9, 86, 61)]


class FaceRemoval(unittest.TestCase):
    def test_face_and_neck_removed(self):
        img, sk = face_scene(), SkinModel()
        without = extract_hand(img, sk)
        with_face = extract_hand(img, sk, faces=FACE)
        self.assertLess(without.bbox[1], 40, "ฉากทดสอบต้องมีใบหน้าเมื่อไม่ตัด")
        self.assertTrue(with_face.found)
        x1, y1, _, _ = with_face.bbox
        self.assertTrue(x1 >= 130 and y1 >= 80, f"ยังรวมใบหน้า/คอ: {with_face.bbox}")

    def test_hand_near_face_kept(self):
        img = face_scene()
        img[100:130, 100:125] = SKIN
        h = extract_hand(img, SkinModel(), faces=FACE)
        self.assertTrue(h.found and h.bbox[0] <= 105, h.bbox)


class WristCut(unittest.TestCase):
    def test_arm_from_border_is_trimmed(self):
        img = blank()
        img[40:90, 70:130] = SKIN          # ฝ่ามือ
        img[90:160, 90:110] = SKIN         # แขนแคบ ติดขอบล่าง
        h = extract_hand(img, SkinModel())
        self.assertTrue(h.found and h.bbox[3] < 125, f"ยังมีแขนยาวเกิน: {h.bbox}")

    def test_no_cut_when_not_touching_border(self):
        img = blank()
        img[40:90, 70:130] = SKIN
        img[90:130, 90:110] = SKIN         # แขนสั้น ไม่ถึงขอบ
        h = extract_hand(img, SkinModel())
        self.assertGreaterEqual(h.bbox[3], 128, h.bbox)


class MotionSelection(unittest.TestCase):
    @staticmethod
    def four_blobs():
        """3 กลุ่มใหญ่นิ่ง + 1 กลุ่มเล็ก (ล่างซ้าย) — ปกติเก็บแค่ 3 กลุ่มที่ใหญ่สุด"""
        img = blank()
        img[5:55, 5:65] = SKIN             # A 3000
        img[5:55, 80:130] = SKIN           # B 2500
        img[5:55, 145:185] = SKIN          # C 2000
        img[90:140, 5:35] = SKIN           # D 1500 (เล็กสุด)
        mov = np.zeros((160, 200), bool)
        mov[90:140, 5:35] = True
        return img, mov

    def test_moving_blob_outranks_larger_static_ones(self):
        img, mov = self.four_blobs()
        sk = SkinModel()
        self.assertLess(extract_hand(img, sk).bbox[3], 60, "ไม่มี motion: กลุ่ม D ต้องถูกตัดทิ้ง")
        self.assertGreaterEqual(extract_hand(img, sk, motion=mov).bbox[3], 140,
                                "มี motion: กลุ่ม D ต้องได้อันดับขึ้นมาและถูกเก็บ")

    def test_static_face_ignored_when_motion_required(self):
        img, sk = face_scene(), SkinModel()
        mov = np.zeros((160, 200), bool)
        self.assertFalse(extract_hand(img, sk, motion=mov, require_motion=True).found)
        mov[90:140, 140:180] = True                                 # มือขยับ
        h = extract_hand(img, sk, motion=mov, require_motion=True)
        self.assertTrue(h.found and h.bbox[0] >= 130, h.bbox)
        still = extract_hand(img, sk, motion=np.zeros_like(mov), require_motion=True, track=h.mask)
        self.assertTrue(still.found and still.bbox[0] >= 130, "ต้องติดตามมือที่หยุดนิ่งต่อได้")


if __name__ == "__main__":
    unittest.main(verbosity=2)
