"""
dataset.py — ขั้น Data Collection / Labeling / Augmentation / Split ของวงจร ML

แหล่งข้อมูล
  1. Raw Data/*.jpg              — รูปต้นแบบ (label + ROI ใน data/annotations.json)
  2. Raw Data/<label>/*.jpg      — (ถ้ามี) รูปเพิ่มเติมแยกโฟลเดอร์ตามปาง
  3. data/collected/<label>/*.png — ตัวอย่างที่เก็บจากกล้องผ่านโปรแกรม app.py
"""
import json
import math
from dataclasses import dataclass

import numpy as np

from . import config
from .features import extract_features
from .image_ops import (box_blur, downscale_area, roi_working_shape, to_uint8,
                        warp_roi)
from .io_utils import read_image_rgb
from .segmentation import extract_hand


@dataclass
class SourceImage:
    path: object
    label: str
    roi: tuple
    kind: str                  # "raw" | "collected"
    exclude: tuple = ()        # กรอบที่ต้องลบทิ้ง (เช่น คอ/คาง ที่ติดกับนิ้ว) — Data cleaning
    image: np.ndarray = None   # RGB uint8 (โหลดภายหลัง)

    def load(self):
        if self.image is None:
            img = read_image_rgb(self.path)
            for x1, y1, x2, y2 in self.exclude:
                img[int(y1):int(y2), int(x1):int(x2)] = 127      # ทาสีเทากลาง = ไม่ใช่สีผิว
            self.image = img
            if self.roi is None:
                h, w = self.image.shape[:2]
                self.roi = (0, 0, w, h)
        return self.image


# ======================================================================
# Data collection + labeling
# ======================================================================
def load_mudra_info():
    return json.loads(config.INFO_FILE.read_text(encoding="utf-8"))


def class_names_from_info():
    """ลำดับคลาสใช้ตามไฟล์คลังความรู้ (mudra_info.json)"""
    return list(load_mudra_info()["mudras"].keys())


def load_sources(include_collected=True):
    ann = json.loads(config.ANNOTATION_FILE.read_text(encoding="utf-8"))
    sources, annotated = [], set()
    for it in ann["images"]:
        p = config.RAW_DIR / it["file"]
        if p.exists():
            sources.append(SourceImage(p, it["label"], tuple(it["roi"]), "raw",
                                       tuple(tuple(b) for b in it.get("exclude", []))))
            annotated.add(p.name)
    if config.RAW_DIR.exists():
        for p in sorted(config.RAW_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS and p.name not in annotated:
                sources.append(SourceImage(p, p.stem, None, "raw"))       # ไม่มี ROI -> ใช้ทั้งภาพ
            elif p.is_dir():
                for q in sorted(p.iterdir()):
                    if q.suffix.lower() in config.IMAGE_EXTS:
                        sources.append(SourceImage(q, p.name, None, "raw"))
    if include_collected and config.COLLECTED_DIR.exists():
        for d in sorted(config.COLLECTED_DIR.iterdir()):
            if d.is_dir():
                for q in sorted(d.iterdir()):
                    if q.suffix.lower() in config.IMAGE_EXTS:
                        sources.append(SourceImage(q, d.name, None, "collected"))
    return sources


# ======================================================================
# Preprocessing: ตัด ROI -> ภาพทำงาน
# ======================================================================
def prepare_source(img, roi, work_size=config.WORK_SIZE):
    """ย่อภาพด้วยตัวคูณจำนวนเต็ม (ลด aliasing) ให้ ROI ใหญ่กว่าภาพทำงานไม่เกิน ~2 เท่า"""
    x1, y1, x2, y2 = roi
    f = max(1, int(max(x2 - x1, y2 - y1) // (work_size * 1.25)))
    return downscale_area(img, f), (x1 / f, y1 / f, x2 / f, y2 / f)


def make_working(img_small, roi_small, aug=None, work_size=config.WORK_SIZE):
    """ตัด ROI ออกมาเป็นภาพทำงาน (ด้านยาว = work_size) พร้อม augmentation (ถ้ามี)"""
    roi = roi_small
    if aug is None:
        h, w = roi_working_shape(roi, work_size)
        return warp_roi(img_small, roi, h, w)
    x1, y1, x2, y2 = roi
    rw, rh = x2 - x1, y2 - y1
    j = aug["roi_jitter"]
    roi = (x1 + j[0] * rw, y1 + j[1] * rh, x2 + j[2] * rw, y2 + j[3] * rh)
    h, w = roi_working_shape(roi, work_size)
    return warp_roi(img_small, roi, h, w, aug["angle"], aug["scale"], aug["flip"], aug["shift"])


# ======================================================================
# Data augmentation (สร้างข้อมูลเพิ่มจากภาพเดิม)
# ======================================================================
def random_augmentation(rng, strength=1.0):
    """สุ่มพารามิเตอร์การแปลงเชิงเรขาคณิต + แสง/สี"""
    s = strength
    return {
        "angle": float(rng.uniform(-18, 18) * s),
        "scale": float(math.exp(rng.uniform(-0.12, 0.12) * s)),
        "flip": bool(rng.random() < 0.5),              # มือซ้าย/ขวา + กล้องกลับด้าน
        "shift": tuple(rng.uniform(-0.06, 0.06, 2) * s),
        "roi_jitter": tuple(rng.uniform(-0.08, 0.08, 4) * s),
        "brightness": float(rng.uniform(0.65, 1.3)),
        "contrast": float(rng.uniform(0.75, 1.25)),
        "channel_gain": tuple(rng.uniform(0.9, 1.1, 3)),
        "gamma": float(math.exp(rng.uniform(-0.25, 0.25))),
        "noise": float(rng.uniform(0, 7)),
        "blur": int(rng.random() < 0.3),
    }


def photometric(work, aug, rng):
    """เปลี่ยนแสง/สี/ความคมชัด/สัญญาณรบกวน เลียนแบบกล้องและสภาพแสงที่ต่างกัน"""
    x = work.astype(np.float32)
    mean = x.mean()
    x = (x - mean) * aug["contrast"] + mean
    x = x * aug["brightness"] * np.asarray(aug["channel_gain"], np.float32)
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** aug["gamma"]
    if aug["blur"]:
        x = box_blur(x, 1)
    if aug["noise"] > 0:
        x = x + rng.normal(0, aug["noise"], x.shape).astype(np.float32)
    return np.clip(x, 0, 255)


# ======================================================================
# สร้างชุดข้อมูลฟีเจอร์
# ======================================================================
def build_feature_dataset(sources, class_names, skin_model, aug_raw=config.AUG_PER_RAW_IMAGE,
                          aug_collected=config.AUG_PER_COLLECTED, seed=config.SEED, log=print):
    """
    คืนค่า dict:
      X (n,d) ฟีเจอร์, y (n,) ป้าย, src (n,) ดัชนีภาพต้นทาง, is_orig (n,) เป็นภาพจริงไม่ผ่าน aug
    """
    rng = np.random.default_rng(seed)
    cls_idx = {c: i for i, c in enumerate(class_names)}
    X, y, src, is_orig = [], [], [], []
    skipped = 0
    for si, s in enumerate(sources):
        if s.label not in cls_idx:
            log(f"  [ข้าม] ไม่รู้จักป้าย '{s.label}' ({s.path.name})")
            continue
        img = s.load()
        small, roi = prepare_source(img, s.roi)
        n_aug = aug_raw if s.kind == "raw" else aug_collected
        ok = 0
        for k in range(n_aug + 1):           # k = 0 คือภาพจริง (ไม่ augment)
            if k == 0:
                work = make_working(small, roi)
            else:
                aug = random_augmentation(rng)
                work = photometric(make_working(small, roi, aug), aug, rng)
            hand = extract_hand(work, skin_model)
            if not hand.found:
                skipped += 1
                continue
            X.append(extract_features(hand))
            y.append(cls_idx[s.label])
            src.append(si)
            is_orig.append(k == 0)
            ok += 1
        log(f"  {s.kind:9s} {s.label:16s} {s.path.name:28s} -> {ok} ตัวอย่าง")
    if skipped:
        log(f"  (ข้าม {skipped} ตัวอย่างที่หามือไม่เจอหลัง augment)")
    return {"X": np.asarray(X, np.float32), "y": np.asarray(y, np.int64),
            "src": np.asarray(src, np.int64), "is_orig": np.asarray(is_orig, bool)}


def split_dataset(ds, sources, ratios=config.SPLIT, seed=config.SEED):
    """
    แบ่ง train / val / test แบบแบ่งชั้นตามคลาส (stratified)
      - ภาพดิบ (มีคลาสละ 1 ภาพ): แบ่งระดับ "ตัวอย่างที่ augment แล้ว"
      - ภาพจากกล้อง: แบ่งระดับ "ภาพต้นทาง" ทั้งก้อน เพื่อกันข้อมูลรั่ว (data leakage)
    """
    rng = np.random.default_rng(seed)
    y, src = ds["y"], ds["src"]
    part = np.full(len(y), -1, np.int64)       # 0=train 1=val 2=test
    kinds = np.array([sources[i].kind for i in src]) if len(src) else np.array([])

    def assign(ids):
        ids = rng.permutation(ids)
        n = len(ids)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        if n >= 3:
            n_tr = min(max(n_tr, 1), n - 2)
            n_va = max(n_va, 1)
        return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]

    for c in np.unique(y):
        raw_ids = np.nonzero((y == c) & (kinds == "raw"))[0]
        for p, ids in enumerate(assign(raw_ids)):
            part[ids] = p
        col_src = np.unique(src[(y == c) & (kinds == "collected")])
        if len(col_src) >= 3:
            for p, sids in enumerate(assign(col_src)):
                part[np.isin(src, sids)] = p
        else:
            part[np.isin(src, col_src)] = 0

    return {name: np.nonzero(part == p)[0] for p, name in enumerate(["train", "val", "test"])}


# ======================================================================
# ภาพตัวอย่างสำหรับ EDA (ดูผลการแบ่งส่วนมือ)
# ======================================================================
def render_segmentation_preview(sources, skin_model, tile=160):
    """
    สร้างภาพตาราง: [ภาพ ROI | mask ผิว | ภาพมือที่ crop | silhouette] หนึ่งแถวต่อหนึ่งภาพต้นทาง
    """
    from .image_ops import resize_bilinear, resize_nearest, draw_rect
    rows = []
    for s in sources:
        img = s.load()
        small, roi = prepare_source(img, s.roi)
        work = make_working(small, roi)
        hand = extract_hand(work, skin_model)
        h, w = work.shape[:2]
        sc = tile / max(h, w)
        th, tw = int(h * sc), int(w * sc)

        def pad(im):
            out = np.full((tile, tile, 3), 40, np.uint8)
            out[:im.shape[0], :im.shape[1]] = im
            return out

        a = to_uint8(resize_bilinear(work, th, tw))
        m = hand.mask if hand.mask is not None else np.zeros((h, w), bool)
        b = resize_nearest(m, th, tw)
        over = a.copy()
        over[b] = (0.5 * over[b] + [0, 120, 0]).astype(np.uint8)
        if hand.found:
            x1, y1, x2, y2 = hand.bbox
            draw_rect(over, (x1 * sc, y1 * sc, x2 * sc, y2 * sc), (255, 60, 60), 2)
            c = to_uint8(np.repeat(resize_nearest(hand.crop_gray, tile, tile)[..., None], 3, 2) * 255)
            d = to_uint8(np.repeat(resize_nearest(hand.crop_mask, tile, tile)[..., None], 3, 2) * 255)
        else:
            c = d = np.zeros((tile, tile, 3), np.uint8)
        gap = np.full((tile, 4, 3), 255, np.uint8)
        rows.append(np.concatenate([pad(a), gap, pad(over), gap, c, gap, d], axis=1))
        rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    return np.concatenate(rows, axis=0)
