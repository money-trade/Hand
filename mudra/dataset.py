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
# ภาพ "ไม่อยู่ในฐานข้อมูล" (unknown) — ใช้ทดสอบว่าระบบปฏิเสธได้จริง (ไม่ใช้ตั้งเกณฑ์)
# ======================================================================
# กรอบบริเวณคอ/คางในภาพดิบ (ไม่มีมืออยู่ในกรอบ) — จำลองกรณีใบหน้า/คอเข้ามาในกรอบ ROI
NECK_ROIS = {"abhaya.jpg": (430, 0, 900, 200), "mudra_R.jpg": (450, 0, 880, 140),
             "chin.jpg": (520, 0, 960, 150)}


def synthetic_blob(rng, size=config.WORK_SIZE):
    """รูปทรงสีผิวแบบสุ่ม: ฝ่ามือวงรี + นิ้วสุ่มจำนวน/ทิศทาง (เช่น แบมือ กางนิ้ว กำมือ) บนพื้นเทา"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.full((size, size, 3), rng.uniform(60, 200), np.float32)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7, 2) * size
    a, b = rng.uniform(0.11, 0.27, 2) * size
    t = rng.uniform(0, np.pi)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1
    for _ in range(int(rng.integers(0, 6))):                      # นิ้ว = แท่งยาวมีความหนา
        phi = rng.uniform(0, 2 * np.pi)
        x0, y0 = cx + 0.8 * a * np.cos(phi), cy + 0.8 * b * np.sin(phi)
        length = rng.uniform(0.12, 0.35) * size
        dx, dy = length * np.cos(phi), length * np.sin(phi)
        s = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (length * length), 0, 1)
        dist2 = (xx - (x0 + s * dx)) ** 2 + (yy - (y0 + s * dy)) ** 2
        mask |= dist2 <= (rng.uniform(0.03, 0.07) * size) ** 2
    skin = np.array([rng.uniform(170, 230), rng.uniform(120, 165), rng.uniform(90, 135)], np.float32)
    shade = 1 + 0.08 * rng.normal(0, 1, (size, size)).astype(np.float32)
    img[mask] = skin * shade[mask][:, None]
    return np.clip(img, 0, 255)


def make_unknown_features(sources, skin_model, n_per_type=60, seed=config.SEED):
    """
    สร้างฟีเจอร์ของภาพที่ "ไม่ใช่ปางใดในฐานข้อมูล" 4 แบบ
      neck    : คอ/คาง (ใบหน้าเข้ามาในกรอบ)
      partial : เห็นมือไม่ครบ — เห็นแต่ข้อมือ/แขน หรือซูมเห็นแค่ส่วนของนิ้ว
      blob    : รูปทรงสีผิวสุ่ม (ท่ามือแปลก ๆ แบมือ กางนิ้ว)
      mix     : ครึ่งซ้ายจากปางหนึ่ง + ครึ่งขวาจากอีกปาง (ท่าที่ไม่มีอยู่จริง ใกล้เคียงข้อมูลมาก)
    """
    from .image_ops import resize_bilinear
    rng = np.random.default_rng(seed)
    raw = [s for s in sources if s.kind == "raw" and s.roi is not None]
    out = {k: [] for k in ("neck", "partial", "blob", "mix")}

    def add(kind, work):
        hand = extract_hand(work, skin_model)
        if hand.found:
            out[kind].append(extract_features(hand))

    neck_src = []
    for s in raw:
        if s.path.name in NECK_ROIS:                         # ใช้ภาพต้นฉบับ (ไม่ลบส่วนคอ)
            img = read_image_rgb(s.path)
            neck_src.append(prepare_source(img, NECK_ROIS[s.path.name]))
    for i in range(n_per_type):
        if neck_src:
            small, roi = neck_src[i % len(neck_src)]
            aug = random_augmentation(rng, 0.6)
            add("neck", photometric(make_working(small, roi, aug), aug, rng))

        s = raw[i % len(raw)]
        small, (x1, y1, x2, y2) = prepare_source(s.load(), s.roi)
        w, h = x2 - x1, y2 - y1
        if rng.random() < 0.5:                                # เลื่อนกรอบลงล่าง: เห็นแต่ข้อมือ/แขน
            dy = rng.uniform(0.55, 0.75) * h
            roi = (x1, y1 + dy, x2, y2 + dy)
        else:                                                 # ซูมเข้าไปเห็นแค่บางส่วนของมือ
            f = rng.uniform(0.3, 0.45)
            px, py = x1 + rng.uniform(0, 1 - f) * w, y1 + rng.uniform(0, 1 - f) * h
            roi = (px, py, px + f * w, py + f * h)
        aug = random_augmentation(rng, 0.6)
        add("partial", photometric(make_working(small, roi, aug), aug, rng))

        add("blob", synthetic_blob(rng))

        a, b = rng.choice(len(raw), 2, replace=False)
        halves = []
        for src in (raw[a], raw[b]):
            sm, r = prepare_source(src.load(), src.roi)
            aug = random_augmentation(rng, 0.6)
            halves.append(resize_bilinear(photometric(make_working(sm, r, aug), aug, rng),
                                          config.WORK_SIZE, config.WORK_SIZE))
        half = config.WORK_SIZE // 2
        add("mix", np.concatenate([halves[0][:, :half], halves[1][:, half:]], axis=1))

    return {k: np.asarray(v, np.float32) for k, v in out.items() if v}


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
