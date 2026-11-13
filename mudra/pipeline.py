"""
pipeline.py — รวมทุกขั้นตอนตอนใช้งานจริง (Inference / Deployment)

  เฟรมกล้อง + ROI -> ภาพทำงาน -> หามือ
     -> [ด่าน 1] ตรวจขนาดมือ (เล็กเกิน / สีผิวเต็มกรอบ)         ไม่ผ่าน = ไม่คำนวณต่อ
     -> สกัดฟีเจอร์ -> Standardize -> PCA
     -> [ด่าน 2] Open-set: อยู่ในฐานข้อมูลหรือไม่ (novelty)    ไม่ผ่าน = ไม่ส่งให้โมเดลจำแนก
     -> โมเดลจำแนก -> ความน่าจะเป็นของแต่ละปาง

ใช้ฟังก์ชันชุดเดียวกับตอนเทรน (dataset.prepare_source / make_working) เพื่อให้การเตรียมข้อมูลตรงกัน 100%
"""
import numpy as np

from . import config
from .dataset import make_working, prepare_source
from .features import extract_features
from .io_utils import detect_faces
from .image_ops import rgb_to_gray
from .models import KNNClassifier, NeuralNetClassifier
from .novelty import NoveltyDetector
from .preprocessing import PCA, StandardScaler
from .segmentation import SkinModel, extract_hand

# สถานะผลวิเคราะห์ 1 เฟรม
NO_HAND = "no_hand"            # ไม่พบสีผิว/มือในกรอบ
BAD_GEOMETRY = "bad_geometry"  # พบ แต่ขนาดไม่เหมาะ (ด่าน 1)
UNKNOWN = "unknown"            # พบมือ แต่ไม่อยู่ในฐานข้อมูล (ด่าน 2)
KNOWN = "known"                # อยู่ในฐานข้อมูล -> จำแนกปางแล้ว


def save_model(path, class_names, scaler, pca, model, model_type, novelty=None):
    """บันทึกทุกอย่างที่ต้องใช้ตอนทำนายลงไฟล์ .npz ไฟล์เดียว"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "class_names": np.array(class_names),
        "model_type": np.array(model_type),
        "scaler_mean": scaler.mean_, "scaler_std": scaler.std_,
        "pca_mean": pca.mean_, "pca_components": pca.components_,
        "pca_scale": np.array(pca.scale_),
        "pca_var_ratio": pca.explained_variance_ratio_,
        "cfg": np.array([config.WORK_SIZE, config.CROP_SIZE, config.HOG_CELL,
                         config.HOG_BINS, config.HOG_BLOCK, config.MASK_GRID]),
    }
    for k, v in model.get_params().items():
        data["m_" + k] = v
    if novelty is not None:
        for k, v in novelty.get_params().items():
            data["nov_" + k] = v
    with open(path, "wb") as f:
        np.savez_compressed(f, **data)


def check_geometry(hand):
    """ด่าน 1: ตรวจขนาดมือก่อนสกัดฟีเจอร์ — คืนข้อความเหตุผลถ้าไม่ผ่าน, None ถ้าผ่าน"""
    if hand.skin_ratio > config.MAX_SKIN_RATIO:
        return "สีผิวเต็มกรอบ — ถอยห่างจากกล้อง หรือย้ายกรอบไม่ให้ใบหน้าอยู่ในกรอบ"
    x1, y1, x2, y2 = hand.bbox
    if max(x2 - x1, y2 - y1) < config.MIN_HAND_SIDE:
        return "มือเล็กเกินไป — ขยับเข้าใกล้กล้อง หรือลากกรอบ ROI ให้เล็กลง"
    return None


class MudraRecognizer:
    def __init__(self, model_path=config.MODEL_FILE, skin_model=None):
        with np.load(model_path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        self.class_names = [str(c) for c in d["class_names"]]
        self.model_type = str(d["model_type"])
        self.scaler = StandardScaler()
        self.scaler.mean_, self.scaler.std_ = d["scaler_mean"], d["scaler_std"]
        self.pca = PCA()
        self.pca.mean_, self.pca.components_ = d["pca_mean"], d["pca_components"]
        self.pca.scale_ = float(d["pca_scale"])
        params = {k[2:]: v for k, v in d.items() if k.startswith("m_")}
        if self.model_type == "knn":
            self.model = KNNClassifier.from_params(params)
        else:
            self.model = NeuralNetClassifier.from_params(params)
        nov = {k[4:]: v for k, v in d.items() if k.startswith("nov_")}
        self.novelty = NoveltyDetector.from_params(nov) if nov else None   # โมเดลเก่าไม่มีด่านนี้
        self._tick, self._faces, self._face_age = 0, [], 0
        self._prev_gray = None
        self._track, self._lost = None, 0
        self.require_motion = False     # True = จับเฉพาะสีผิวที่ขยับ/ติดตามอยู่ (ใช้กับกล้องสด)
        self.ignore_faces = True        # ตัดใบหน้าออกก่อนหามือ
        self.sensitivity = 1.0          # ตัวคูณเกณฑ์: > 1 ยอมรับง่ายขึ้น, < 1 เข้มงวดขึ้น
        self.skin_model = skin_model or SkinModel.load(config.SKIN_FILE)

    def transform(self, F):
        """ฟีเจอร์ -> (Z ในปริภูมิ PCA, reconstruction error)"""
        Xs = self.scaler.transform(np.atleast_2d(F))
        return self.pca.transform(Xs), self.pca.reconstruction_error(Xs)

    def predict_proba_features(self, F):
        return self.model.predict_proba(self.transform(F)[0])

    def work_faces(self, frame_rgb, roi, work_shape):
        """ตำแหน่งใบหน้าในพิกัดภาพทำงาน (ตรวจทุก FACE_EVERY เฟรม และจำผลไว้ FACE_HOLD เฟรม กันพลาดตอนหันหน้า)"""
        if not self.ignore_faces:
            return []
        self._tick += 1
        if self._tick % config.FACE_EVERY == 1 or self._tick == 1:
            try:
                found = detect_faces(frame_rgb)
            except Exception:
                found = []
            if found:
                self._faces, self._face_age = found, 0
            else:
                self._face_age += config.FACE_EVERY
        if self._face_age > config.FACE_HOLD:
            self._faces = []
        x1, y1, x2, y2 = roi
        sx, sy = work_shape[1] / max(1.0, x2 - x1), work_shape[0] / max(1.0, y2 - y1)
        return [((fx1 - x1) * sx, (fy1 - y1) * sy, (fx2 - x1) * sx, (fy2 - y1) * sy)
                for fx1, fy1, fx2, fy2 in self._faces]

    def work_motion(self, work):
        """mask การเคลื่อนไหวบนภาพทำงาน (ต่างจากเฟรมก่อน) — ไม่มีเฟรมก่อน/ขนาดเปลี่ยน = None"""
        g = rgb_to_gray(work)
        prev, self._prev_gray = self._prev_gray, g
        if prev is None or prev.shape != g.shape:
            return None
        m = np.abs(g - prev) > config.MOTION_DIFF
        return m if m.mean() < 0.5 else None            # กล้องสั่น/แสงเปลี่ยนทั้งภาพ = ไม่ใช้

    def analyze(self, frame_rgb, roi):
        """
        วิเคราะห์ 1 เฟรม  roi = (x1, y1, x2, y2) ในพิกัดของ frame_rgb
        คืน dict: status (no_hand | bad_geometry | unknown | known), reason, novelty,
                  probs/label/confidence (มีค่าเฉพาะ status = known), hand, work
        """
        small, roi_s = prepare_source(frame_rgb, roi)
        work = make_working(small, roi_s)
        faces = self.work_faces(frame_rgb, roi, work.shape)             # ตัดใบหน้า โฟกัสแค่มือ
        motion = self.work_motion(work)
        hand = extract_hand(work, self.skin_model, faces=faces, motion=motion,
                            require_motion=self.require_motion, track=self._track)
        if self.require_motion:                                         # ติดตามมือที่พบล่าสุด
            if hand.found:
                self._track, self._lost = hand.mask, 0
            else:
                self._lost += 1
                if self._lost > config.HAND_LOST_FRAMES:
                    self._track = None
        out = {"status": NO_HAND, "found": hand.found, "hand": hand, "work": work, "reason": "",
               "novelty": None, "probs": None, "label": None, "confidence": 0.0}
        if not hand.found:
            return out

        reason = check_geometry(hand)                                   # ด่าน 1
        if reason:
            out.update(status=BAD_GEOMETRY, reason=reason)
            return out

        Z, E = self.transform(extract_features(hand))
        if self.novelty is not None:                                    # ด่าน 2
            score = float(self.novelty.scores(Z, E)[0])
            out["novelty"] = score
            if score > self.sensitivity:
                out.update(status=UNKNOWN, reason="ท่ามือนี้ไม่ตรงกับปางใดในฐานข้อมูล")
                return out                                              # ไม่ส่งให้โมเดลจำแนก

        probs = self.model.predict_proba(Z)[0]                          # จำแนกปาง
        i = int(probs.argmax())
        out.update(status=KNOWN, probs=probs, label=self.class_names[i], confidence=float(probs[i]))
        return out
