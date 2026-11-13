"""
image_ops.py — ฟังก์ชันประมวลผลภาพพื้นฐาน เขียนเองทั้งหมดด้วย NumPy
(ไม่เรียกใช้ฟังก์ชันประมวลผลภาพของ OpenCV / PIL / scikit-image)

ภาพสี ใช้รูปแบบ RGB, shape = (H, W, 3)
ภาพเทา / mask ใช้ shape = (H, W)
"""
import math

import numpy as np


# ======================================================================
# แปลงปริภูมิสี
# ======================================================================
def rgb_to_gray(img):
    """แปลง RGB -> ความสว่าง (มาตรฐาน ITU-R BT.601)"""
    img = img.astype(np.float32)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def rgb_to_ycrcb(img):
    """แปลง RGB -> (Y, Cr, Cb)  ใช้แยกสีผิว เพราะสีผิวจับกลุ่มกันดีในช่อง Cr/Cb"""
    img = img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


# ======================================================================
# ปรับขนาด / สุ่มตัวอย่างพิกเซล
# ======================================================================
def downscale_area(img, f):
    """ย่อภาพด้วยตัวคูณจำนวนเต็ม f โดยเฉลี่ยพิกเซลในบล็อก f x f (ลด aliasing)"""
    img = img.astype(np.float32)
    if f <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // f, w // f
    img = img[: h2 * f, : w2 * f]
    if img.ndim == 3:
        return img.reshape(h2, f, w2, f, img.shape[2]).mean(axis=(1, 3))
    return img.reshape(h2, f, w2, f).mean(axis=(1, 3))


def sample_bilinear(img, xs, ys, fill=127.0):
    """
    อ่านค่าพิกเซลที่พิกัดทศนิยม (xs, ys) ด้วย bilinear interpolation
    พิกัดที่อยู่นอกภาพจะได้ค่า fill
    """
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)
    x1, y1 = x0 + 1, y0 + 1
    valid = (xs >= -0.5) & (xs <= w - 0.5) & (ys >= -0.5) & (ys <= h - 0.5)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    out = (img[y0c, x0c] * (1 - wx) * (1 - wy) + img[y0c, x1c] * wx * (1 - wy)
           + img[y1c, x0c] * (1 - wx) * wy + img[y1c, x1c] * wx * wy)
    out[~valid] = fill
    return out


def resize_bilinear(img, out_h, out_w):
    """ปรับขนาดภาพด้วย bilinear (จัดกึ่งกลางพิกเซลให้ตรงกัน)"""
    h, w = img.shape[:2]
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    gx, gy = np.meshgrid(np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1))
    return sample_bilinear(img, gx, gy)


def resize_nearest(img, out_h, out_w):
    """ปรับขนาดแบบ nearest neighbour (ใช้กับ mask หรือภาพแสดงผล)"""
    h, w = img.shape[:2]
    yi = np.minimum((np.arange(out_h) * h / out_h).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(out_w) * w / out_w).astype(np.int64), w - 1)
    return img[yi][:, xi]


def roi_working_shape(roi, work_size):
    """คำนวณขนาดภาพทำงาน (h, w) ให้ด้านยาว = work_size และคงอัตราส่วนของ ROI"""
    x1, y1, x2, y2 = roi
    rw, rh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if rw >= rh:
        return max(8, int(round(work_size * rh / rw))), work_size
    return work_size, max(8, int(round(work_size * rw / rh)))


def warp_roi(img, roi, out_h, out_w, angle_deg=0.0, scale=1.0, flip=False,
             shift=(0.0, 0.0), fill=127.0):
    """
    ตัดภาพในกรอบ roi=(x1,y1,x2,y2) ออกมาเป็นขนาด out_h x out_w
    พร้อมแปลงแบบ affine (หมุน / ย่อขยาย / พลิก / เลื่อน) ในขั้นตอนเดียว
    - ใช้ทั้งตอน "เพิ่มข้อมูล" (augmentation) และตอนใช้งานจริง (ไม่หมุน ไม่พลิก)
    - การหมุนอ้างอิงจุดกึ่งกลางของ ROI และดึงพิกเซลจากภาพเต็ม จึงไม่เกิดขอบดำ
    """
    x1, y1, x2, y2 = roi
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx, sy = (x2 - x1) / out_w, (y2 - y1) / out_h
    u = np.arange(out_w, dtype=np.float32) - (out_w - 1) / 2.0
    v = np.arange(out_h, dtype=np.float32) - (out_h - 1) / 2.0
    gu, gv = np.meshgrid(u, v)
    if flip:
        gu = -gu
    gu, gv = gu * sx / scale, gv * sy / scale
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    xs = c * gu - s * gv + cx + shift[0] * (x2 - x1)
    ys = s * gu + c * gv + cy + shift[1] * (y2 - y1)
    return sample_bilinear(img, xs, ys, fill=fill)


# ======================================================================
# ฟิลเตอร์
# ======================================================================
def _box1d(a, r, axis):
    """ค่าเฉลี่ยเคลื่อนที่ 1 มิติ ด้วยผลรวมสะสม (cumulative sum) — O(n) ไม่ขึ้นกับ r"""
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def box_blur(img, r=1):
    """เบลอด้วยกล่อง (2r+1)x(2r+1) แบบแยกแกน (separable)"""
    if r <= 0:
        return img.astype(np.float32)
    return _box1d(_box1d(img.astype(np.float32), r, 0), r, 1)


# ======================================================================
# Morphology บน mask (bool)
# ======================================================================
def _shift_reduce(mask, r, axis, op, pad_mode):
    n = mask.shape[axis]
    pad = [(0, 0)] * mask.ndim
    pad[axis] = (r, r)
    p = np.pad(mask, pad, mode=pad_mode)
    out = None
    for k in range(2 * r + 1):
        sl = [slice(None)] * mask.ndim
        sl[axis] = slice(k, k + n)
        part = p[tuple(sl)]
        out = part.copy() if out is None else op(out, part)
    return out


def dilate(mask, r=1):
    """ขยาย (dilation) ด้วยโครงสร้างสี่เหลี่ยม (2r+1)^2"""
    if r <= 0:
        return mask.copy()
    m = _shift_reduce(mask, r, 0, np.logical_or, "constant")
    return _shift_reduce(m, r, 1, np.logical_or, "constant")


def erode(mask, r=1):
    """กร่อน (erosion) — ขอบภาพใช้ค่าซ้ำขอบ เพื่อไม่กัดแขนที่ล้นกรอบ ROI"""
    if r <= 0:
        return mask.copy()
    m = _shift_reduce(mask, r, 0, np.logical_and, "edge")
    return _shift_reduce(m, r, 1, np.logical_and, "edge")


def opening(mask, r=1):
    """erode แล้ว dilate — ลบจุดรบกวนเล็ก ๆ"""
    return dilate(erode(mask, r), r)


def closing(mask, r=1):
    """dilate แล้ว erode — อุดรูเล็ก ๆ / เชื่อมรอยแตก"""
    return erode(dilate(mask, r), r)


# ======================================================================
# Connected components (ติดป้ายกลุ่มพิกเซลที่เชื่อมต่อกัน)
# ======================================================================
def connected_components(mask, connectivity=4):
    """
    ติดป้ายกลุ่มพิกเซลที่ติดกันด้วยวิธี label propagation + pointer jumping
      1) ให้ทุกพิกเซล foreground มีป้าย = ดัชนีของตัวเอง
      2) แต่ละรอบ: ป้าย = ค่าน้อยสุดของตัวเองกับเพื่อนบ้าน
      3) pointer jumping: ป้าย = ป้ายของพิกเซลที่ป้ายชี้ไป (ลู่เข้าเร็วแบบ log)
    คืนค่า (labels[H,W] เริ่มที่ 1, จำนวนกลุ่ม, พื้นที่ของแต่ละกลุ่ม)
    """
    h, w = mask.shape
    n = h * w
    fg = mask.ravel().astype(bool)
    if not fg.any():
        return np.zeros((h, w), np.int32), 0, np.zeros(0, np.int64)
    inf = n
    lab = np.where(fg, np.arange(n), inf).astype(np.int64)
    for _ in range(4 * (h + w)):
        grid = lab.reshape(h, w)
        p = np.pad(grid, 1, constant_values=inf)
        m = np.minimum(grid, p[:-2, 1:-1])
        m = np.minimum(m, p[2:, 1:-1])
        m = np.minimum(m, p[1:-1, :-2])
        m = np.minimum(m, p[1:-1, 2:])
        if connectivity == 8:
            m = np.minimum(m, p[:-2, :-2])
            m = np.minimum(m, p[:-2, 2:])
            m = np.minimum(m, p[2:, :-2])
            m = np.minimum(m, p[2:, 2:])
        new = np.where(fg, m.ravel(), inf)
        for _ in range(3):
            new[fg] = new[new[fg]]
        if np.array_equal(new, lab):
            break
        lab = new
    uniq, inv = np.unique(lab[fg], return_inverse=True)
    labels = np.zeros(n, np.int32)
    labels[fg] = inv.ravel() + 1
    areas = np.bincount(inv.ravel(), minlength=len(uniq))
    return labels.reshape(h, w), len(uniq), areas


def count_holes(mask, min_area=6):
    """
    นับ "รู" ในรูปทรงมือ (พื้นหลังที่ถูกล้อมรอบ เช่น วงกลมของนิ้วโป้ง-นิ้วชี้)
    คืนค่า (จำนวนรู, พื้นที่รูรวม)
    """
    bg_labels, n, areas = connected_components(~mask, connectivity=4)
    if n == 0:
        return 0, 0
    border = np.unique(np.concatenate([bg_labels[0], bg_labels[-1],
                                       bg_labels[:, 0], bg_labels[:, -1]]))
    n_holes, hole_area = 0, 0
    for i in range(1, n + 1):
        if i in border:
            continue
        a = int(areas[i - 1])
        if a >= min_area:
            n_holes += 1
            hole_area += a
    return n_holes, hole_area


# ======================================================================
# วาดภาพแบบง่าย (สำหรับรายงาน / ภาพตัวอย่าง)
# ======================================================================
def draw_rect(img, box, color, thickness=2):
    """วาดกรอบสี่เหลี่ยมลงบนภาพ RGB (แก้ไขภาพในตัว)"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    t = thickness
    img[y1:y1 + t, x1:x2 + 1] = color
    img[max(y2 - t + 1, 0):y2 + 1, x1:x2 + 1] = color
    img[y1:y2 + 1, x1:x1 + t] = color
    img[y1:y2 + 1, max(x2 - t + 1, 0):x2 + 1] = color
    return img


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)
