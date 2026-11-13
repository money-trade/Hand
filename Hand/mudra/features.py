"""
features.py — สกัดฟีเจอร์ (Feature Extraction) จากภาพมือ เขียนเองทั้งหมด

เวกเตอร์ฟีเจอร์ 1 ตัวอย่าง ประกอบด้วย
  1. HOG (Histogram of Oriented Gradients) — ทิศทางขอบ/เส้นนิ้ว     1764 มิติ
  2. Silhouette grid — รูปทรงมือแบบหยาบ 16x16                       256 มิติ
  3. Hu moments — โมเมนต์ที่ไม่แปรตามการหมุน/ย่อขยาย                  7 มิติ
  4. Shape descriptors — สัดส่วนกรอบ ความทึบ จำนวนมือ รู ทิศแกนหลัก     8 มิติ
"""
import math

import numpy as np

from . import config


def hog(gray, cell=config.HOG_CELL, bins=config.HOG_BINS, block=config.HOG_BLOCK):
    """
    HOG แบบมาตรฐาน (Dalal & Triggs 2005)
      - gradient แบบ central difference
      - ทิศทาง 0..180 องศา แบ่ง `bins` ช่อง กระจายน้ำหนักแบบ linear ระหว่างช่องที่ติดกัน
      - รวมเป็น histogram ต่อ cell แล้ว normalize ต่อ block ด้วย L2-Hys
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0

    bw = 180.0 / bins
    pos = ang / bw - 0.5
    b0 = np.floor(pos).astype(np.int64)
    w1 = pos - b0
    b0 = b0 % bins
    b1 = (b0 + 1) % bins

    ch, cw = h // cell, w // cell
    cy = np.minimum(np.arange(h) // cell, ch - 1)
    cx = np.minimum(np.arange(w) // cell, cw - 1)
    cell_idx = (cy[:, None] * cw + cx[None, :])
    size = ch * cw * bins
    hist = (np.bincount((cell_idx * bins + b0).ravel(), (mag * (1 - w1)).ravel(), size)
            + np.bincount((cell_idx * bins + b1).ravel(), (mag * w1).ravel(), size))
    hist = hist.reshape(ch, cw, bins)

    nby, nbx = ch - block + 1, cw - block + 1
    blocks = np.concatenate([hist[i:i + nby, j:j + nbx]
                             for i in range(block) for j in range(block)], axis=-1)
    eps = 1e-6
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps)
    blocks = np.minimum(blocks, 0.2)                                   # L2-Hys: clip
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps)
    return blocks.ravel().astype(np.float32)


def mask_grid(mask, grid=config.MASK_GRID):
    """ย่อ mask ด้วยการเฉลี่ยบล็อก -> รูปทรงมือหยาบ ๆ"""
    h, w = mask.shape
    fy, fx = h // grid, w // grid
    return mask[: fy * grid, : fx * grid].reshape(grid, fy, grid, fx).mean(axis=(1, 3)).ravel()


def _central_moments(mask):
    h, w = mask.shape
    m = mask.astype(np.float64)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    m00 = m.sum()
    if m00 <= 1e-9:
        return None
    xc, yc = (m * xs).sum() / m00, (m * ys).sum() / m00
    dx, dy = xs - xc, ys - yc

    def mu(p, q):
        return (m * dx ** p * dy ** q).sum()

    return m00, mu


def hu_moments(mask):
    """Hu's 7 invariant moments (1962) + log scale ให้อยู่ช่วงตัวเลขที่เหมาะ"""
    res = _central_moments(mask)
    if res is None:
        return np.zeros(7, np.float32)
    m00, mu = res

    def eta(p, q):
        return mu(p, q) / m00 ** (1 + (p + q) / 2.0)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
          + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h6 = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
          + 4 * n11 * (n30 + n12) * (n21 + n03))
    h7 = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
          - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu = np.array([h1, h2, h3, h4, h5, h6, h7])
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30) / 10.0).astype(np.float32)


def shape_descriptors(hand):
    """คุณลักษณะรูปทรงเชิงเรขาคณิตของมือ"""
    x1, y1, x2, y2 = hand.bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    area = float(hand.mask[y1:y2, x1:x2].sum())
    fill = area / (bw * bh)

    # ทิศทางแกนหลัก + ความรี (จาก second-order central moments)
    res = _central_moments(hand.crop_mask > 0.5)
    orient_c, orient_s, ecc = 0.0, 0.0, 0.0
    if res is not None:
        m00, mu = res
        a, b, c = mu(2, 0) / m00, mu(1, 1) / m00, mu(0, 2) / m00
        theta = 0.5 * math.atan2(2 * b, a - c)
        common = math.sqrt(((a - c) / 2) ** 2 + b * b)
        l1, l2 = (a + c) / 2 + common, (a + c) / 2 - common
        ecc = math.sqrt(max(0.0, 1 - l2 / l1)) if l1 > 1e-9 else 0.0
        orient_c, orient_s = math.cos(2 * theta), math.sin(2 * theta)

    return np.array([
        math.log(bw / bh),
        fill,
        min(hand.n_components, 3) / 3.0,
        min(hand.n_holes, 3) / 3.0,
        min(hand.hole_ratio * 10.0, 1.0),
        orient_c, orient_s, ecc,
    ], np.float32)


def extract_features(hand):
    """รวมฟีเจอร์ทั้งหมดเป็นเวกเตอร์เดียว"""
    return np.concatenate([
        hog(hand.crop_gray),
        mask_grid(hand.crop_mask),
        hu_moments(hand.crop_mask),
        shape_descriptors(hand),
    ]).astype(np.float32)


def feature_dim():
    cells = config.CROP_SIZE // config.HOG_CELL
    nb = cells - config.HOG_BLOCK + 1
    return nb * nb * config.HOG_BLOCK ** 2 * config.HOG_BINS + config.MASK_GRID ** 2 + 7 + 8
