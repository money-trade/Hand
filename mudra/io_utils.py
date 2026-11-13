"""
io_utils.py — งาน Input/Output เท่านั้น (ส่วนเดียวในโปรเจคที่เรียกใช้ OpenCV)

OpenCV ถูกใช้แค่ 3 อย่าง:
  1. ถอดรหัสไฟล์ JPG/PNG เป็นอาร์เรย์พิกเซล       (cv2.imdecode)
  2. เข้ารหัสอาร์เรย์พิกเซลเป็นไฟล์ PNG           (cv2.imencode)
  3. อ่านเฟรมจากกล้องเว็บแคม                      (cv2.VideoCapture)
ไม่มีการใช้ฟังก์ชันประมวลผลภาพหรือ Machine Learning ของ OpenCV เลย
"""
import os
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # ให้ส่วนอื่นของโปรแกรม import ได้แม้ยังไม่ติดตั้ง
    cv2 = None


def _need_cv2():
    if cv2 is None:
        raise RuntimeError("ต้องติดตั้ง opencv-python ก่อน (pip install opencv-python) — ใช้เปิดกล้องและอ่านไฟล์รูปเท่านั้น")


def read_image_rgb(path):
    """อ่านไฟล์รูปเป็น RGB uint8 (รองรับพาธภาษาไทยบน Windows)"""
    _need_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"อ่านไฟล์รูปไม่ได้: {path}")
    return np.ascontiguousarray(img[..., ::-1])


def write_image_rgb(path, rgb):
    """บันทึกอาร์เรย์ RGB เป็นไฟล์ (นามสกุลกำหนดชนิดไฟล์)"""
    _need_cv2()
    path = str(path)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., ::-1]))
    if not ok:
        raise IOError(f"บันทึกไฟล์ไม่ได้: {path}")
    buf.tofile(path)


_FACE = None


def detect_faces(rgb, max_width=320):
    """
    หาตำแหน่งใบหน้า (Haar cascade ที่มากับ OpenCV) เพื่อ "ตัดใบหน้าออก" ไม่ให้ถูกนับเป็นมือ
    คืน list ของ (x1, y1, x2, y2) ในพิกัดของ rgb — ถ้าใช้ไม่ได้คืน list ว่าง
    """
    global _FACE
    if cv2 is None:
        return []
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                   "haarcascade_frontalface_default.xml"))
    if _FACE.empty():
        return []
    h, w = rgb.shape[:2]
    s = min(1.0, max_width / float(w))
    small = rgb if s == 1.0 else cv2.resize(rgb, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_RGB2GRAY)
    faces = _FACE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                   minSize=(int(24), int(24)))
    return [(int(x / s), int(y / s), int((x + fw) / s), int((y + fh) / s)) for x, y, fw, fh in faces]


class Camera:
    """อ่านเฟรมจากเว็บแคมใน thread แยก (เก็บเฉพาะเฟรมล่าสุด เพื่อไม่ให้ภาพหน่วง)"""

    def __init__(self, index=0, width=640, height=480):
        _need_cv2()
        self.cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(index, be)
            if cap is not None and cap.isOpened():
                self.cap = cap
                break
            if cap is not None:
                cap.release()
        if self.cap is None:
            raise RuntimeError(f"เปิดกล้องหมายเลข {index} ไม่ได้")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            rgb = np.ascontiguousarray(frame[..., ::-1])
            with self._lock:
                self._frame = rgb

    def read(self):
        """คืนเฟรมล่าสุด (RGB uint8) หรือ None ถ้ายังไม่มี"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class DemoCamera:
    """
    กล้องจำลอง: วนแสดงรูปจาก Raw Data ทีละรูป (ใช้ทดสอบโปรแกรมเมื่อไม่มีเว็บแคม)
    รูปถูกย่อลงกลางเฟรม 640x480 และให้ ROI ตามไฟล์ annotations
    """

    def __init__(self, items, width=640, height=480, seconds_per_image=4.0):
        from .image_ops import resize_bilinear, to_uint8
        self.frames = []
        for img, roi in items:
            h, w = img.shape[:2]
            s = min(width / w, height / h)
            nw, nh = int(w * s), int(h * s)
            small = to_uint8(resize_bilinear(img, nh, nw))
            canvas = np.full((height, width, 3), 200, np.uint8)
            ox, oy = (width - nw) // 2, (height - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = small
            r = [roi[0] * s + ox, roi[1] * s + oy, roi[2] * s + ox, roi[3] * s + oy]
            self.frames.append((canvas, r))
        self.seconds = seconds_per_image
        self.t0 = time.time()

    def index(self):
        return int((time.time() - self.t0) / self.seconds) % len(self.frames)

    def read(self):
        return self.frames[self.index()][0].copy()

    def current_roi(self):
        return self.frames[self.index()][1]

    def release(self):
        pass
