"""
pipeline.py — รวมทุกขั้นตอนตอนใช้งานจริง (Inference / Deployment)

  เฟรมกล้อง + ROI -> ภาพทำงาน -> หามือ -> สกัดฟีเจอร์ -> Standardize -> PCA -> โมเดล -> ความน่าจะเป็น

ใช้ฟังก์ชันชุดเดียวกับตอนเทรน (dataset.prepare_source / make_working) เพื่อให้การเตรียมข้อมูลตรงกัน 100%
"""
import numpy as np

from . import config
from .dataset import make_working, prepare_source
from .features import extract_features
from .models import KNNClassifier, NeuralNetClassifier
from .preprocessing import PCA, StandardScaler
from .segmentation import SkinModel, extract_hand


def save_model(path, class_names, scaler, pca, model, model_type):
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
    with open(path, "wb") as f:
        np.savez_compressed(f, **data)


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
        self.skin_model = skin_model or SkinModel.load(config.SKIN_FILE)

    def transform(self, F):
        return self.pca.transform(self.scaler.transform(np.atleast_2d(F)))

    def predict_proba_features(self, F):
        return self.model.predict_proba(self.transform(F))

    def analyze(self, frame_rgb, roi):
        """
        วิเคราะห์ 1 เฟรม: คืน dict {found, probs, label, confidence, hand, work}
        roi = (x1, y1, x2, y2) ในพิกัดของ frame_rgb
        """
        small, roi_s = prepare_source(frame_rgb, roi)
        work = make_working(small, roi_s)
        hand = extract_hand(work, self.skin_model)
        out = {"found": hand.found, "hand": hand, "work": work, "probs": None,
               "label": None, "confidence": 0.0}
        if not hand.found:
            return out
        probs = self.predict_proba_features(extract_features(hand))[0]
        i = int(probs.argmax())
        out.update(probs=probs, label=self.class_names[i], confidence=float(probs[i]))
        return out
