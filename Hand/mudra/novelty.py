"""
novelty.py — Open-set recognition: ตรวจว่า "ภาพมือนี้มีอยู่ในฐานข้อมูลหรือไม่" ก่อนส่งให้โมเดลจำแนก

ปัญหา
  โมเดลจำแนก (softmax) ถูกบังคับให้เลือก 1 ใน 7 ปางเสมอ แม้ภาพจะไม่ใช่ปางใดเลย
  (เช่น กำมือ แบมือ ใบหน้า แขน) และมักให้ความมั่นใจสูงเกินจริง จึงใช้ความมั่นใจตัดสินไม่ได้

วิธีแก้ (เขียนเองทั้งหมด ไม่ต้องใช้ตัวอย่าง "ภาพที่ไม่รู้จัก" ในการเรียนรู้ จึงใช้ได้กับท่าแปลกใหม่ทุกแบบ)
  วัดความห่างจากข้อมูลที่รู้จัก 2 มุมมอง ในปริภูมิ PCA
    1. d = ระยะ kNN ถึง "กลุ่มของปางที่ใกล้ที่สุด"
           d_c = ระยะถึงตัวอย่างฝึกของปาง c ลำดับที่ k ÷ ระยะปกติของปาง c   (ปางที่กระจายมากได้เกณฑ์กว้าง)
           d   = min_c d_c                     -> ไม่ต้องรันโมเดลจำแนกก่อน
    2. r = reconstruction error ของ PCA ÷ ค่าปกติ
           -> ภาพที่มีลักษณะที่ไม่เคยเห็นตอนเทรน จะสร้างกลับจากแกนหลักได้ไม่ดี
  novelty score = sqrt(d × r) ÷ T
    <= 1  -> อยู่ในฐานข้อมูล  -> ส่งให้โมเดลจำแนกต่อ
    >  1  -> ไม่อยู่ในฐานข้อมูล -> ไม่นำไปคำนวณ รอภาพถัดไป
  เกณฑ์ T = เปอร์เซ็นไทล์ที่ 90 ของคะแนนภาพที่รู้จัก (validation + ภาพบิดแรง ×1.5)
"""
import numpy as np

from . import config


def _kth_distance(R, Q, k):
    """ระยะ Euclidean จากแต่ละแถวของ Q ถึงเพื่อนบ้านลำดับที่ k ใน R  (||a-b||² = ||a||² + ||b||² - 2a·b)"""
    d2 = (Q * Q).sum(axis=1)[:, None] + (R * R).sum(axis=1)[None, :] - 2.0 * (Q @ R.T)
    k = min(k, R.shape[0])
    return np.sqrt(np.maximum(np.partition(d2, k - 1, axis=1)[:, k - 1], 0.0))


class NoveltyDetector:
    def __init__(self, k=config.NOVELTY_K, max_refs=config.NOVELTY_MAX_REFS, seed=0):
        self.k = k
        self.max_refs = max_refs
        self.seed = seed
        self.thr_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, Z_train, y_train, Z_val, y_val, E_val):
        """
        เก็บตัวอย่างอ้างอิงแยกตามปาง + เรียนรู้ "ระยะปกติ" ของแต่ละปางจากชุด validation
        """
        Z = np.asarray(Z_train, np.float32)
        y = np.asarray(y_train, np.int64)
        if len(Z) > self.max_refs:
            idx = np.random.default_rng(self.seed).choice(len(Z), self.max_refs, replace=False)
            Z, y = Z[idx], y[idx]
        self.refs_, self.ref_y_ = Z, y
        self.classes_ = np.unique(y)
        Z_val, y_val = np.asarray(Z_val, np.float32), np.asarray(y_val)
        overall = []
        scale = {}
        for c in self.classes_:
            m = y_val == c
            if m.any():
                scale[int(c)] = float(np.median(_kth_distance(self.refs_[y == c], Z_val[m], self.k)))
                overall.append(scale[int(c)])
        fallback = float(np.median(overall)) if overall else 1.0
        self.class_scale_ = np.array([max(scale.get(int(c), fallback), 1e-6) for c in self.classes_])
        self.recon_scale_ = float(max(np.median(E_val), 1e-9))
        return self

    def components(self, Z, E):
        """คืน (d = ระยะถึงกลุ่มปางที่ใกล้สุดแบบปรับสเกล, r = reconstruction error แบบปรับสเกล)"""
        Z = np.atleast_2d(np.asarray(Z, np.float32))
        D = np.stack([_kth_distance(self.refs_[self.ref_y_ == c], Z, self.k) / s
                      for c, s in zip(self.classes_, self.class_scale_)], axis=1)
        return D.min(axis=1), np.atleast_1d(np.asarray(E, np.float32)) / self.recon_scale_

    def raw_score(self, Z, E):
        d, r = self.components(Z, E)
        return np.sqrt(d * r)

    def calibrate(self, Z_cal, E_cal, percentile=config.NOVELTY_PERCENTILE,
                  margin=config.NOVELTY_MARGIN):
        """ตั้งเกณฑ์จากภาพที่รู้จักเท่านั้น (ภาพ unknown ใช้ทดสอบอย่างเดียว ไม่ให้รู้ข้อสอบล่วงหน้า)"""
        self.thr_ = float(np.percentile(self.raw_score(Z_cal, E_cal), percentile) * margin)
        return self

    # ------------------------------------------------------------------ score
    def scores(self, Z, E):
        """คะแนนความแปลกใหม่ หน่วยเป็น "เท่าของเกณฑ์"  (<= 1 คืออยู่ในฐานข้อมูล)"""
        return self.raw_score(Z, E) / self.thr_

    def is_known(self, Z, E, sensitivity=1.0):
        """sensitivity > 1 ยอมรับง่ายขึ้น, < 1 เข้มงวดขึ้น"""
        return self.scores(Z, E) <= sensitivity

    # ------------------------------------------------------------------ save / load
    def get_params(self):
        return {"refs": self.refs_, "ref_y": self.ref_y_, "classes": self.classes_,
                "class_scale": self.class_scale_, "recon_scale": np.array(self.recon_scale_),
                "k": np.array(self.k), "thr": np.array(self.thr_)}

    @staticmethod
    def from_params(p):
        d = NoveltyDetector(int(p["k"]))
        d.refs_ = p["refs"].astype(np.float32)
        d.ref_y_ = p["ref_y"].astype(np.int64)
        d.classes_ = p["classes"].astype(np.int64)
        d.class_scale_ = p["class_scale"].astype(np.float64)
        d.recon_scale_ = float(p["recon_scale"])
        d.thr_ = float(p["thr"])
        return d
