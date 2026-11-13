"""
segmentation.py — ตรวจหาบริเวณมือจากสีผิว (Skin segmentation) เขียนเองทั้งหมด

ขั้นตอน
  ภาพ ROI -> เบลอเล็กน้อย -> แปลงเป็น YCrCb -> แยกพิกเซลสีผิว
  -> morphology (opening/closing) -> connected components (เก็บกลุ่มใหญ่ = มือ/แขน)
  -> หา bounding box -> crop เป็นสี่เหลี่ยมจัตุรัส -> ปรับขนาดเป็น CROP_SIZE
"""
import json
from dataclasses import dataclass, field

import numpy as np

from . import config
from .image_ops import (box_blur, closing, connected_components, count_holes,
                        opening, rgb_to_gray, rgb_to_ycrcb, sample_bilinear)


class SkinModel:
    """
    โมเดลสีผิว 2 แบบ
      - "threshold": กำหนดช่วง Cr, Cb ตายตัว (ค่าเริ่มต้น ใช้ได้กับคนส่วนใหญ่)
      - "gaussian" : เรียนรู้ค่าเฉลี่ยและความแปรปรวนร่วมของสีผิวผู้ใช้ (ปรับเทียบจากกล้อง)
                     แล้วใช้ระยะ Mahalanobis ตัดสินว่าเป็นผิวหรือไม่
    """

    def __init__(self, mode="threshold", cr=config.SKIN_CR, cb=config.SKIN_CB,
                 y_min=config.SKIN_Y_MIN, mean=None, icov=None, max_dist=3.0):
        self.mode = mode
        self.cr, self.cb, self.y_min = tuple(cr), tuple(cb), float(y_min)
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.icov = None if icov is None else np.asarray(icov, np.float32)
        self.max_dist = float(max_dist)

    def mask(self, rgb):
        y, cr, cb = rgb_to_ycrcb(rgb)
        if self.mode == "gaussian" and self.mean is not None:
            d0, d1 = cr - self.mean[0], cb - self.mean[1]
            a, b, c = self.icov[0, 0], self.icov[0, 1], self.icov[1, 1]
            dist2 = a * d0 * d0 + 2 * b * d0 * d1 + c * d1 * d1
            return (dist2 <= self.max_dist ** 2) & (y >= self.y_min)
        return ((cr >= self.cr[0]) & (cr <= self.cr[1]) &
                (cb >= self.cb[0]) & (cb <= self.cb[1]) & (y >= self.y_min))

    @staticmethod
    def fit_gaussian(rgb_pixels, max_dist=3.0):
        """เรียนรู้โมเดลสีผิวแบบเกาส์เซียน 2 มิติ (Cr, Cb) จากพิกเซลตัวอย่าง (N,3)"""
        px = np.asarray(rgb_pixels, np.float32).reshape(-1, 3)
        y, cr, cb = rgb_to_ycrcb(px)
        keep = y > 30
        x = np.stack([cr[keep], cb[keep]], axis=1)
        if len(x) < 20:
            raise ValueError("พิกเซลตัวอย่างน้อยเกินไป")
        mean = x.mean(axis=0)
        d = x - mean
        cov = d.T @ d / (len(x) - 1) + np.eye(2, dtype=np.float32) * 4.0  # regularize
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        icov = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det
        y_min = max(20.0, float(np.percentile(y[keep], 2)) * 0.5)
        return SkinModel("gaussian", mean=mean, icov=icov, y_min=y_min, max_dist=max_dist)

    def to_dict(self):
        return {"mode": self.mode, "cr": list(self.cr), "cb": list(self.cb), "y_min": self.y_min,
                "mean": None if self.mean is None else self.mean.tolist(),
                "icov": None if self.icov is None else self.icov.tolist(),
                "max_dist": self.max_dist}

    @staticmethod
    def from_dict(d):
        return SkinModel(d.get("mode", "threshold"), d.get("cr", config.SKIN_CR),
                         d.get("cb", config.SKIN_CB), d.get("y_min", config.SKIN_Y_MIN),
                         d.get("mean"), d.get("icov"), d.get("max_dist", 3.0))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path):
        if path.exists():
            return SkinModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return SkinModel()


@dataclass
class HandResult:
    """ผลลัพธ์การหามือในภาพ ROI"""
    found: bool
    mask: np.ndarray = None                 # mask ผิวบนภาพทำงาน (bool)
    crop_gray: np.ndarray = None            # ภาพเทาของมือ CROP x CROP (0..1, พื้นหลัง = 0)
    crop_mask: np.ndarray = None            # mask ของมือ CROP x CROP (0..1)
    bbox: tuple = None                      # กรอบมือบนภาพทำงาน (x1,y1,x2,y2)
    skin_ratio: float = 0.0
    n_components: int = 0
    n_holes: int = 0
    hole_ratio: float = 0.0
    extra: dict = field(default_factory=dict)


def remove_faces(mask, faces):
    """
    ตัดใบหน้าออกจาก mask สีผิว (faces = กรอบใบหน้า x1,y1,x2,y2 ในพิกัดภาพทำงาน)
      1) ลบพิกเซลสีผิวในกรอบใบหน้า (เฉพาะสีผิว — มือที่ไม่ใช่สีผิวหรืออยู่นอกกรอบไม่ถูกแตะ)
      2) ลบกลุ่มสีผิวที่ติดกับใบหน้าและมีศูนย์กลางอยู่ในโซนหู/คอ (ส่วนของหัวที่เหลือ)
    มือที่อยู่ไกลจากหน้าจะไม่ถูกลบ
    """
    if not faces:
        return mask
    h, w = mask.shape
    mask = mask.copy()
    for x1, y1, x2, y2 in faces:
        fw, fh = x2 - x1, y2 - y1
        cx0, cx1 = int(max(0, x1 - 0.08 * fw)), int(min(w, x2 + 0.08 * fw))
        cy0, cy1 = int(max(0, y1 - 0.15 * fh)), int(min(h, y2 + 0.08 * fh))
        mask[cy0:cy1, cx0:cx1] = False
        labels, n, areas = connected_components(mask)
        if n == 0:
            continue
        # โซนหู/คอ: กว้างกว่าหน้าเล็กน้อย ลงไปใต้คางประมาณ 0.8 ของความสูงหน้า
        zx0, zx1 = x1 - 0.3 * fw, x2 + 0.3 * fw
        zy0, zy1 = y1 - 0.4 * fh, y2 + 0.8 * fh
        ring = np.zeros_like(mask)
        ring[max(0, cy0 - 3):min(h, cy1 + 3), max(0, cx0 - 3):min(w, cx1 + 3)] = True
        ring[cy0:cy1, cx0:cx1] = False
        for lab in np.unique(labels[ring & (labels > 0)]):
            ys, xs = np.nonzero(labels == lab)
            if zx0 <= xs.mean() <= zx1 and zy0 <= ys.mean() <= zy1:
                mask[labels == lab] = False
    return mask


def extract_hand(work_rgb, skin_model, crop_size=config.CROP_SIZE,
                 min_skin_ratio=config.MIN_SKIN_RATIO, faces=None):
    """หาบริเวณมือในภาพทำงาน (ภาพ ROI ที่ปรับขนาดแล้ว) แล้ว crop ให้พร้อมสกัดฟีเจอร์"""
    h, w = work_rgb.shape[:2]
    smooth = box_blur(work_rgb, 1)
    raw_mask = skin_model.mask(smooth)
    mask = closing(opening(raw_mask, 1), 2)
    mask = remove_faces(mask, faces)

    labels, n, areas = connected_components(mask)
    if n == 0:
        return HandResult(False, mask=mask)

    # เก็บเฉพาะกลุ่มใหญ่ (มือ/แขน) สูงสุด 3 กลุ่ม — ตัดสิ่งรบกวนสีคล้ายผิว
    order = np.argsort(-areas)
    largest = areas[order[0]]
    keep = [i + 1 for i in order[:3]
            if areas[i] >= max(0.15 * largest, 0.004 * h * w)]
    hand = np.isin(labels, keep)
    ratio = float(hand.mean())
    if ratio < min_skin_ratio:
        return HandResult(False, mask=hand, skin_ratio=ratio)

    ys, xs = np.nonzero(hand)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x2 - x1, y2 - y1) * 1.08
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # crop สี่เหลี่ยมจัตุรัสรอบมือ (คงอัตราส่วนเดิม) แล้วย่อเป็น crop_size
    g = (np.arange(crop_size, dtype=np.float32) + 0.5) / crop_size - 0.5
    gx, gy = np.meshgrid(cx + g * side - 0.5, cy + g * side - 0.5)
    crop_mask = sample_bilinear(hand.astype(np.float32), gx, gy, fill=0.0)
    gray = rgb_to_gray(work_rgb) / 255.0
    crop_gray = sample_bilinear(gray, gx, gy, fill=0.0)

    # ปรับความสว่างภายในมือให้อยู่ช่วง 0..1 (ลดผลจากแสงต่างกัน) แล้วลบพื้นหลังทิ้ง
    m = crop_mask > 0.5
    if m.sum() > 10:
        vals = crop_gray[m]
        lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
        crop_gray = np.clip((crop_gray - lo) / max(hi - lo, 1e-3), 0, 1) * 0.8 + 0.2
    crop_gray = crop_gray * crop_mask

    n_holes, hole_area = count_holes(m, min_area=6)
    return HandResult(True, mask=hand, crop_gray=crop_gray.astype(np.float32),
                      crop_mask=crop_mask.astype(np.float32),
                      bbox=(int(x1), int(y1), int(x2), int(y2)), skin_ratio=ratio,
                      n_components=len(keep), n_holes=n_holes,
                      hole_ratio=hole_area / max(1.0, float(m.sum())))
