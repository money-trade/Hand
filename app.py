"""
app.py — โปรแกรมหลัก: ตรวจจับท่ามือจากกล้อง แล้วแปลงเป็นข้อความ "ปางพระพุทธรูป"

หน้าจอ
  ซ้าย : ภาพกล้อง + กรอบ ROI + สีผิวที่ตรวจพบ (เขียว) + การเคลื่อนไหว (แดง)
         + ภาพที่โมเดลเห็น + มาตรวัดการเคลื่อนไหว
  ขวา  : ชื่อปาง / ความมั่นใจ / ความหมาย / ความน่าจะเป็นทุกปาง / ข้อความที่ถอดได้ต่อเนื่อง

วิธีใช้
    python app.py            # ใช้เว็บแคม
    python app.py --demo     # ไม่มีกล้อง: วนภาพจาก Raw Data แทน

คีย์ลัด
    1-7  เก็บตัวอย่างของปางนั้นจากกรอบ ROI      C  ล้างข้อความ     S  บันทึกข้อความ
    ลากเมาส์บนภาพกล้อง = กำหนดกรอบ ROI ใหม่      Q / Esc  ออก
"""
import argparse
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import messagebox, ttk

import numpy as np

from mudra import config
from mudra.dataset import class_names_from_info, load_mudra_info, load_sources
from mudra.image_ops import resize_nearest, rgb_to_gray
from mudra.nlp import NONE_TOKEN, HMMFilter, ThaiTextGenerator, TranscriptBuilder
from mudra.segmentation import SkinModel

W, H = 640, 480

C_BG = "#17181c"
C_PANEL = "#23252c"
C_CARD = "#2c2f38"
C_TEXT = "#f1f1f1"
C_MUTED = "#a3a8b4"
C_GOLD = "#e3b04b"
C_GREEN = "#4cd07d"
C_YELLOW = "#f2c94c"
C_RED = "#eb5757"


def to_photo(rgb):
    """แปลงอาร์เรย์ RGB -> tk.PhotoImage ผ่านรูปแบบ PPM (ไม่ต้องใช้ PIL)"""
    h, w = rgb.shape[:2]
    return tk.PhotoImage(data=f"P6 {w} {h} 255 ".encode() + np.ascontiguousarray(rgb, np.uint8).tobytes(),
                         format="PPM")


class MotionDetector:
    """
    ตรวจจับการเคลื่อนไหวด้วย Background subtraction
      background_t = (1-a) * background_{t-1} + a * frame_t   (running average)
      พิกเซลเคลื่อนไหว = |frame_t - background_{t-1}| > threshold
    """

    def __init__(self, alpha=0.25, threshold=config.MOTION_PIXEL_THRESHOLD, step=4):
        self.alpha, self.threshold, self.step = alpha, threshold, step
        self.bg = None

    def update(self, frame):
        g = rgb_to_gray(frame[::self.step, ::self.step])
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g
            return np.zeros(g.shape, bool)
        moving = np.abs(g - self.bg) > self.threshold
        self.bg = self.bg * (1 - self.alpha) + g * self.alpha
        return moving


class MudraApp:
    def __init__(self, root, camera, demo=False, mirror=True, max_ticks=None):
        self.root, self.cam, self.demo, self.mirror = root, camera, demo, mirror
        self.max_ticks = max_ticks
        self.info = load_mudra_info()
        self.nlg = ThaiTextGenerator(self.info)
        self.skin = SkinModel.load(config.SKIN_FILE)
        self.recognizer = None
        self.class_names = class_names_from_info()
        self.load_model()

        self.motion = MotionDetector()
        self.motion_ema = 0.0
        self.roi = [170, 175, 470, 470]
        self.drag_start = None
        self.msgs = queue.Queue()
        self.training = False
        self.burst_left, self.burst_label, self.burst_wait = 0, None, 0
        self.calib_until = None
        self.last_frame = None
        self.ticks, self.fps, self.t_last = 0, 0.0, time.time()
        self.prev_desc = None

        # ย่อภาพกล้องบนจอให้พอดีความสูงหน้าจอ (การประมวลผลยังใช้ 640x480 เสมอ)
        avail = root.winfo_screenheight() - 80
        self.s = max(0.6, min(1.0, (avail - 300) / H))
        self.compact = self.s < 0.95
        self.DW, self.DH = int(W * self.s), int(H * self.s)
        self.PV = 96 if self.compact else 104

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.refresh_counts()
        self.tick()

    # ================================================================== model
    def load_model(self):
        from mudra.pipeline import MudraRecognizer
        if config.MODEL_FILE.exists():
            try:
                self.recognizer = MudraRecognizer(skin_model=self.skin)
                self.class_names = self.recognizer.class_names
            except Exception as e:  # ไฟล์เสีย / เวอร์ชันไม่ตรง
                self.recognizer = None
                print("โหลดโมเดลไม่สำเร็จ:", e)
        self.hmm = HMMFilter(len(self.class_names))
        old = getattr(self, "transcript", None)          # เทรนใหม่แล้วไม่ลบข้อความเดิม
        if old is None or old.class_names != list(self.class_names):
            self.transcript = TranscriptBuilder(self.class_names)

    # ================================================================== UI
    def _build_ui(self):
        r = self.root
        r.title("Mudra → Text : ระบบตรวจจับปางพระพุทธรูปจากท่ามือ")
        r.configure(bg=C_BG)
        fams = set(tkfont.families())
        fam = next((f for f in ("Leelawadee UI", "Leelawadee", "Tahoma") if f in fams), "TkDefaultFont")
        k = -2 if self.compact else 0
        self.fam = fam
        self.h1_size = 24 + 2 * k
        self.f_h1 = (fam, self.h1_size, "bold")
        self.f_h2 = (fam, 13 + k)
        self.f_hd = (fam, 12 + k // 2, "bold")
        self.f_body = (fam, 11 + k // 2)
        self.f_small = (fam, 9)
        self.row_h = 18 if self.compact else 20

        top = tk.Frame(r, bg=C_BG)
        top.pack(fill="x", padx=14, pady=(6, 2))
        tk.Label(top, text="☸  ระบบแปลท่ามือเป็นปางพระพุทธรูป", font=(fam, 17, "bold"),
                 fg=C_GOLD, bg=C_BG).pack(side="left")
        mode = "โหมดสาธิต (ภาพจาก Raw Data)" if self.demo else "กล้องเว็บแคม"
        tk.Label(top, text=f"ML + NLP เขียนเองทั้งหมด  •  {mode}", font=self.f_small,
                 fg=C_MUTED, bg=C_BG).pack(side="right")

        main = tk.Frame(r, bg=C_BG)
        main.pack(fill="both", expand=True, padx=14, pady=4)

        # ---------------- ซ้าย: กล้อง
        left = tk.Frame(main, bg=C_BG)
        left.pack(side="left", fill="y")
        self.cv = tk.Canvas(left, width=self.DW, height=self.DH, bg="black", highlightthickness=0)
        self.cv.pack()
        self.cv_img = self.cv.create_image(0, 0, anchor="nw")
        self.cv_roi = self.cv.create_rectangle(0, 0, 1, 1, outline=C_YELLOW, width=3)
        self.cv_box = self.cv.create_rectangle(0, 0, 0, 0, outline="#5ab0ff", width=2, state="hidden")
        self.cv_calib = self.cv.create_rectangle(0, 0, 0, 0, outline="white", width=2, dash=(4, 3),
                                                 state="hidden")
        self.cv_tag_bg = self.cv.create_rectangle(0, 0, 0, 0, fill="#000000", outline="", stipple="gray50")
        self.cv_tag = self.cv.create_text(0, 0, anchor="sw", fill="white", font=self.f_hd)
        self.cv_hint = self.cv.create_text(self.DW // 2, self.DH - 10, anchor="s", fill="white", font=self.f_small,
                                           text="วางมือในกรอบ • ลากเมาส์เพื่อย้ายกรอบ ROI")
        if not self.demo:
            self.cv.bind("<ButtonPress-1>", self._roi_press)
            self.cv.bind("<B1-Motion>", self._roi_drag)
            self.cv.bind("<ButtonRelease-1>", self._roi_release)

        row = tk.Frame(left, bg=C_BG)
        row.pack(fill="x", pady=(8, 0))
        self.pv_crop = self._preview(row, "ภาพที่โมเดลเห็น")
        self.pv_mask = self._preview(row, "รูปทรงมือ (mask)")
        meter = tk.Frame(row, bg=C_PANEL, padx=10, pady=8)
        meter.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(meter, text="การเคลื่อนไหว", font=self.f_hd, fg=C_TEXT, bg=C_PANEL).pack(anchor="w")
        self.cv_motion = tk.Canvas(meter, width=200, height=16, bg=C_CARD, highlightthickness=0)
        self.cv_motion.pack(anchor="w", pady=4)
        self.cv_motion_bar = self.cv_motion.create_rectangle(0, 0, 0, 16, fill=C_GREEN, outline="")
        thr_x = int(200 * min(1.0, config.MOTION_MOVING_RATIO / 0.3))
        self.cv_motion.create_line(thr_x, 0, thr_x, 16, fill="white", dash=(2, 2))
        self.lbl_motion = tk.Label(meter, text="-", font=self.f_body, fg=C_MUTED, bg=C_PANEL,
                                   width=26, anchor="w")
        self.lbl_motion.pack(anchor="w")
        self.lbl_fps = tk.Label(meter, text="FPS -", font=self.f_small, fg=C_MUTED, bg=C_PANEL)
        self.lbl_fps.pack(anchor="w", pady=(6, 0))
        self.lbl_model = tk.Label(meter, text="", font=self.f_small, fg=C_MUTED, bg=C_PANEL,
                                  justify="left", wraplength=210)
        self.lbl_model.pack(anchor="w")

        # ---------------- ขวา: ข้อความ (ทุกป้ายมีความสูงคงที่ หน้าจอจะไม่กระตุกเมื่อข้อความเปลี่ยน)
        right_h = max(self.DH + 150, 520 if self.compact else 640)
        right = tk.Frame(main, bg=C_PANEL, padx=16, pady=10, width=540, height=right_h)
        right.pack(side="left", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        wrap = 500

        tk.Label(right, text="ผลการตรวจจับ", font=self.f_small, fg=C_MUTED, bg=C_PANEL).pack(anchor="w")
        self.lbl_title = tk.Label(right, text="-", font=self.f_h1, fg=C_GOLD, bg=C_PANEL, anchor="w")
        self.lbl_title.pack(anchor="w", fill="x")
        self.lbl_sub = tk.Label(right, text="", font=self.f_h2, fg=C_MUTED, bg=C_PANEL, anchor="w")
        self.lbl_sub.pack(anchor="w", fill="x")
        self.cv_conf = tk.Canvas(right, height=8, bg=C_CARD, highlightthickness=0)
        self.cv_conf.pack(fill="x", pady=(4, 4))
        self.cv_conf_bar = self.cv_conf.create_rectangle(0, 0, 0, 8, fill=C_GOLD, outline="")
        self.lbl_sentence = tk.Label(right, text="", font=self.f_hd, fg=C_TEXT, bg=C_PANEL, height=2,
                                     wraplength=wrap, justify="left", anchor="nw")
        self.lbl_sentence.pack(anchor="w", fill="x")
        self.lbl_hand = tk.Label(right, text="", font=self.f_body, fg=C_TEXT, bg=C_PANEL,
                                 height=1 if self.compact else 2,
                                 wraplength=wrap, justify="left", anchor="nw")
        self.lbl_hand.pack(anchor="w", fill="x", pady=(2, 0))
        self.lbl_meaning = tk.Label(right, text="", font=self.f_body, fg=C_MUTED, bg=C_PANEL,
                                    height=2 if self.compact else 3,
                                    wraplength=wrap, justify="left", anchor="nw")
        self.lbl_meaning.pack(anchor="w", fill="x", pady=(2, 4))

        tk.Label(right, text="ความน่าจะเป็นของแต่ละปาง (หลังกรองด้วย HMM)", font=self.f_small,
                 fg=C_MUTED, bg=C_PANEL).pack(anchor="w")
        n = len(self.class_names)
        rh = self.row_h
        self.cv_probs = tk.Canvas(right, height=rh * n + 4, bg=C_PANEL, highlightthickness=0)
        self.cv_probs.pack(fill="x", pady=(2, 6))
        self.prob_items = []
        for i, c in enumerate(self.class_names):
            y = 2 + i * rh
            t = self.cv_probs.create_text(0, y + rh / 2, anchor="w", text=self.nlg.short(c),
                                          fill=C_TEXT, font=self.f_small)
            bg = self.cv_probs.create_rectangle(110, y + 3, 440, y + rh - 3, fill=C_CARD, outline="")
            bar = self.cv_probs.create_rectangle(110, y + 3, 110, y + rh - 3, fill=C_GOLD, outline="")
            pct = self.cv_probs.create_text(500, y + rh / 2, anchor="e", text="", fill=C_MUTED,
                                            font=self.f_small)
            self.prob_items.append((t, bg, bar, pct))

        tk.Label(right, text="ข้อความที่ถอดได้ (Transcript)", font=self.f_hd, fg=C_TEXT,
                 bg=C_PANEL).pack(anchor="w")
        # สรุปวางชิดล่างก่อน เพื่อไม่ให้ถูกกล่องข้อความเบียดหาย
        self.lbl_summary = tk.Label(right, text=self.nlg.summarize([]), font=self.f_body, fg=C_GOLD,
                                    bg=C_PANEL, wraplength=wrap, justify="left", anchor="nw", height=2)
        self.lbl_summary.pack(side="bottom", anchor="w", fill="x", pady=(4, 0))
        tf = tk.Frame(right, bg=C_PANEL)
        tf.pack(fill="both", expand=True, pady=(2, 0))
        self.txt = tk.Text(tf, height=2, font=self.f_body, bg=C_CARD, fg=C_TEXT, relief="flat",
                           wrap="word", insertbackground=C_TEXT, padx=8, pady=4)
        sb = tk.Scrollbar(tf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)

        # ---------------- ปุ่มควบคุม
        ctl = tk.Frame(r, bg=C_BG)
        ctl.pack(fill="x", padx=14, pady=(6, 2))
        tk.Label(ctl, text="เก็บข้อมูลจริง:", font=self.f_body, fg=C_TEXT, bg=C_BG).pack(side="left")
        self.class_var = tk.StringVar()
        choices = [f"{i + 1}. {self.nlg.short(c)}" for i, c in enumerate(self.class_names)]
        self.cb_class = ttk.Combobox(ctl, values=choices, textvariable=self.class_var, width=16,
                                     state="readonly", font=self.f_body)
        self.cb_class.current(0)
        self.cb_class.pack(side="left", padx=4)
        self.cb_class.bind("<<ComboboxSelected>>", lambda e: self.refresh_counts())
        self.btn_one = self._button(ctl, "เก็บ 1 ภาพ", lambda: self.save_sample(self.selected_label()))
        self.btn_burst = self._button(ctl, "เก็บต่อเนื่อง 20 ภาพ", self.start_burst)
        self.lbl_count = tk.Label(ctl, text="", font=self.f_small, fg=C_MUTED, bg=C_BG)
        self.lbl_count.pack(side="left", padx=6)
        self.btn_train = self._button(ctl, "ฝึกโมเดลใหม่", self.start_training, C_GOLD, "#1b1b1b")
        self._button(ctl, "ปรับเทียบสีผิว", self.start_calibration)
        self._button(ctl, "สีผิวค่าเริ่มต้น", self.reset_skin)
        self._button(ctl, "ล้างข้อความ", self.clear_transcript)
        self._button(ctl, "บันทึกข้อความ", self.save_transcript)

        self.status = tk.Label(r, text="", font=self.f_small, fg=C_MUTED, bg=C_BG, anchor="w")
        self.status.pack(fill="x", padx=14, pady=(0, 8))
        self.set_status("พร้อมใช้งาน" if self.recognizer else
                        "ยังไม่มีโมเดล — กด 'ฝึกโมเดลใหม่' หรือรัน python train.py")
        self._update_model_label()

    def _preview(self, parent, title):
        f = tk.Frame(parent, bg=C_PANEL, padx=6, pady=4)
        f.pack(side="left", padx=(0, 8))
        tk.Label(f, text=title, font=self.f_small, fg=C_MUTED, bg=C_PANEL).pack()
        c = tk.Canvas(f, width=self.PV, height=self.PV, bg="black", highlightthickness=0)
        c.pack()
        c.img_item = c.create_image(0, 0, anchor="nw")
        return c

    def _button(self, parent, text, cmd, bg=C_CARD, fg=C_TEXT):
        b = tk.Button(parent, text=text, command=cmd, font=self.f_small, bg=bg, fg=fg, relief="flat",
                      activebackground=C_GOLD, activeforeground="#111", padx=8, pady=3, cursor="hand2")
        b.pack(side="left", padx=3)
        return b

    def _bind_keys(self):
        for i in range(min(9, len(self.class_names))):
            self.root.bind(str(i + 1), lambda e, i=i: self.save_sample(self.class_names[i]))
        self.root.bind("<c>", lambda e: self.clear_transcript())
        self.root.bind("<s>", lambda e: self.save_transcript())
        self.root.bind("<q>", lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())

    def set_status(self, text):
        self.status.configure(text=text)

    def _update_model_label(self):
        if self.recognizer is None:
            self.lbl_model.configure(text="โมเดล: ยังไม่มี")
            return
        txt = f"โมเดล: {self.recognizer.model.name}"
        if config.MODEL_META_FILE.exists():
            import json
            m = json.loads(config.MODEL_META_FILE.read_text(encoding="utf-8"))
            txt += f"\nเวอร์ชัน {m.get('version', '-')}\ntest acc {m['test']['accuracy'] * 100:.1f}%"
        self.lbl_model.configure(text=txt)

    # ================================================================== ROI ด้วยเมาส์
    def _roi_press(self, e):
        self.drag_start = (e.x, e.y)

    def _roi_drag(self, e):
        if self.drag_start:
            x0, y0 = self.drag_start
            self.cv.coords(self.cv_roi, x0, y0, e.x, e.y)

    def _roi_release(self, e):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        s = self.s                                  # พิกัดบนจอ -> พิกัดภาพ 640x480
        x1, x2 = sorted((max(0, min(W, x0 / s)), max(0, min(W, e.x / s))))
        y1, y2 = sorted((max(0, min(H, y0 / s)), max(0, min(H, e.y / s))))
        if x2 - x1 >= 80 and y2 - y1 >= 80:
            self.roi = [x1, y1, x2, y2]
            self.hmm.reset()
            self.set_status(f"ตั้งกรอบ ROI ใหม่: {self.roi}")

    def current_roi(self):
        return list(self.cam.current_roi()) if self.demo else list(self.roi)

    # ================================================================== main loop
    def tick(self):
        try:
            self._process_messages()
            frame = self.cam.read()
            if frame is not None:
                self._process_frame(frame)
        except Exception as e:  # ไม่ให้โปรแกรมหยุดเพราะเฟรมเดียวผิดพลาด
            self.set_status(f"ข้อผิดพลาด: {e!r}")
            import traceback
            traceback.print_exc()
        self.ticks += 1
        if self.max_ticks and self.ticks >= self.max_ticks:
            self.quit()
            return
        self.root.after(5, self.tick)

    def _process_frame(self, frame):
        if self.mirror and not self.demo:
            frame = np.ascontiguousarray(frame[:, ::-1])
        if frame.shape[:2] != (H, W):
            frame = resize_nearest(frame, H, W)
        self.last_frame = frame
        roi = self.current_roi()
        x1, y1, x2, y2 = [int(round(v)) for v in roi]

        # --- 1) จับการเคลื่อนไหว
        moving = self.motion.update(frame)
        s = self.motion.step
        ratio = float(moving[y1 // s:y2 // s, x1 // s:x2 // s].mean()) if moving.size else 0.0
        self.motion_ema = 0.6 * self.motion_ema + 0.4 * ratio
        still = self.motion_ema < config.MOTION_MOVING_RATIO

        # --- 2) หามือ + จำแนกปาง
        res = self.recognizer.analyze(frame, roi) if self.recognizer else None
        found = bool(res and res["found"])

        # --- 3) NLP: HMM กรองผลตามเวลา -> ถอดเป็นโทเคน -> สร้างประโยค
        if self.recognizer:
            alpha = self.hmm.update(res["probs"] if found else None)
            state = int(alpha.argmax())
            label = self.class_names[state] if state < len(self.class_names) else NONE_TOKEN
            conf = float(alpha[state])
            cls = alpha[:-1] / max(alpha[:-1].sum(), 1e-9)
            order = np.argsort(-cls)
            runner = self.class_names[int(order[1])] if len(order) > 1 else None
            tok = self.transcript.update(label, conf, still, self.hmm.emission(res["probs"] if found else None))
            if tok:
                self._append_token(tok)
            if label == NONE_TOKEN:
                kind = "no_hand"
            elif not still:
                kind = "moving"
            else:
                kind = "ok"
            desc = self.nlg.describe(label, conf, runner, kind)
        else:
            cls, conf, kind = None, 0.0, "no_model"
            desc = self.nlg.describe(None, 0, None, "no_model")

        # --- 4) เก็บข้อมูล / ปรับเทียบสีผิว
        self._handle_burst()
        self._handle_calibration(frame, roi)

        # --- 5) วาดผล
        self._draw_camera(frame, moving, res, roi, kind, still)
        self._draw_previews(res)
        self._draw_panel(desc, cls, conf, kind)
        self._update_fps(ratio, still)

    # ================================================================== drawing
    def _draw_camera(self, frame, moving, res, roi, kind, still):
        disp = frame.copy()
        s = self.motion.step
        mm = moving.repeat(s, 0).repeat(s, 1)[:H, :W]
        if mm.any():
            disp[mm] = (disp[mm] * 0.55 + np.array([255, 70, 70]) * 0.45).astype(np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        hand = res["hand"] if res else None
        if hand is not None and hand.mask is not None and x2 > x1 and y2 > y1:
            m = resize_nearest(hand.mask, y2 - y1, x2 - x1)
            sub = disp[y1:y2, x1:x2]
            sub[m] = (sub[m] * 0.55 + np.array([60, 220, 110]) * 0.45).astype(np.uint8)
        if self.s != 1.0:
            disp = resize_nearest(disp, self.DH, self.DW)
        self.photo = to_photo(disp)
        self.cv.itemconfigure(self.cv_img, image=self.photo)

        s = self.s
        color = {"ok": C_GREEN, "moving": C_YELLOW, "no_hand": C_RED, "no_model": C_MUTED}[kind]
        if not self.drag_start:
            self.cv.coords(self.cv_roi, x1 * s, y1 * s, x2 * s, y2 * s)
        self.cv.itemconfigure(self.cv_roi, outline=color)
        if hand is not None and hand.found:
            wh, ww = res["work"].shape[:2]
            sx, sy = (x2 - x1) / ww, (y2 - y1) / wh
            bx1, by1, bx2, by2 = hand.bbox
            self.cv.coords(self.cv_box, (x1 + bx1 * sx) * s, (y1 + by1 * sy) * s,
                           (x1 + bx2 * sx) * s, (y1 + by2 * sy) * s)
            self.cv.itemconfigure(self.cv_box, state="normal")
        else:
            self.cv.itemconfigure(self.cv_box, state="hidden")
        tag = {"ok": "นิ่ง • กำลังตรวจจับ", "moving": "กำลังเคลื่อนไหว",
               "no_hand": "ไม่พบมือในกรอบ", "no_model": "ยังไม่มีโมเดล"}[kind]
        if kind == "ok" and self.prev_desc:
            tag = self.prev_desc
        self.cv.itemconfigure(self.cv_tag, text=" " + tag + " ")
        self.cv.coords(self.cv_tag, x1 * s, max(22, y1 * s - 4))
        bb = self.cv.bbox(self.cv_tag)
        if bb:
            self.cv.coords(self.cv_tag_bg, *bb)

    def _draw_previews(self, res):
        hand = res["hand"] if res else None
        pv = self.PV
        if hand is not None and hand.found:
            g = (resize_nearest(hand.crop_gray, pv, pv) * 255).astype(np.uint8)
            m = (resize_nearest(hand.crop_mask, pv, pv) * 255).astype(np.uint8)
        else:
            g = m = np.zeros((pv, pv), np.uint8)
        self.ph_crop = to_photo(np.repeat(g[..., None], 3, 2))
        self.ph_mask = to_photo(np.repeat(m[..., None], 3, 2))
        self.pv_crop.itemconfigure(self.pv_crop.img_item, image=self.ph_crop)
        self.pv_mask.itemconfigure(self.pv_mask.img_item, image=self.ph_mask)

    def _draw_panel(self, desc, cls, conf, kind):
        self.prev_desc = desc["title"] if kind == "ok" else None
        color = C_GOLD if kind == "ok" and desc["title"] != "ไม่แน่ใจ" else C_MUTED
        if self.lbl_title.cget("text") != desc["title"]:
            n = len(desc["title"])          # ชื่อยาวใช้ฟอนต์เล็กลง ให้อยู่บรรทัดเดียวเสมอ
            size = self.h1_size if n <= 28 else (self.h1_size - 4 if n <= 36 else self.h1_size - 7)
            self.lbl_title.configure(text=desc["title"], font=(self.fam, size, "bold"))
        self.lbl_title.configure(fg=color)
        self.lbl_sub.configure(text=desc["subtitle"])
        self.lbl_sentence.configure(text=desc["sentence"])
        self.lbl_hand.configure(text=("ลักษณะท่ามือ: " + desc["hand"]) if desc["hand"] else "")
        self.lbl_meaning.configure(text=("ความหมาย: " + desc["meaning"]) if desc["meaning"] else "")
        wbar = self.cv_conf.winfo_width()
        self.cv_conf.coords(self.cv_conf_bar, 0, 0, wbar * (conf if kind in ("ok", "moving") else 0), 8)
        top = int(np.argmax(cls)) if cls is not None and kind in ("ok", "moving") else -1
        rh = self.row_h
        for i, (t, bg, bar, pct) in enumerate(self.prob_items):
            p = float(cls[i]) if cls is not None and kind in ("ok", "moving") else 0.0
            self.cv_probs.coords(bar, 110, 2 + i * rh + 3, 110 + 330 * p, 2 + i * rh + rh - 3)
            self.cv_probs.itemconfigure(bar, fill=C_GOLD if i == top else "#6b6f7a")
            self.cv_probs.itemconfigure(pct, text=f"{p * 100:5.1f}%")

    def _update_fps(self, ratio, still):
        now = time.time()
        dt = now - self.t_last
        self.t_last = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt
        self.lbl_fps.configure(text=f"FPS {self.fps:4.1f}")
        self.cv_motion.coords(self.cv_motion_bar, 0, 0, 200 * min(1.0, self.motion_ema / 0.3), 16)
        self.cv_motion.itemconfigure(self.cv_motion_bar, fill=C_GREEN if still else C_YELLOW)
        self.lbl_motion.configure(text=("นิ่ง" if still else "กำลังเคลื่อนไหว") +
                                  f"  ({self.motion_ema * 100:.0f}% ของกรอบ)")

    # ================================================================== transcript
    def _append_token(self, tok):
        self.txt.insert("end", self.nlg.token_line(tok) + "\n")
        self.txt.see("end")
        self.lbl_summary.configure(text=self.nlg.summarize([t[0] for t in self.transcript.tokens]))

    def clear_transcript(self):
        self.transcript.reset()
        self.txt.delete("1.0", "end")
        self.lbl_summary.configure(text=self.nlg.summarize([]))
        self.set_status("ล้างข้อความแล้ว")

    def save_transcript(self):
        config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRANSCRIPT_DIR / f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"
        live = [t[0] for t in self.transcript.tokens]
        vit = self.transcript.viterbi_tokens(self.hmm.A)
        lines = ["บันทึกข้อความจากระบบแปลท่ามือเป็นปางพระพุทธรูป",
                 f"วันที่ {datetime.now():%Y-%m-%d %H:%M:%S}", "",
                 "== ลำดับท่าที่ตรวจพบ (เวลาจริง) =="]
        lines += [self.nlg.token_line(t) for t in self.transcript.tokens]
        lines += ["", "สรุป: " + self.nlg.summarize(live), "",
                  "== ถอดรหัสทั้งช่วงด้วย Viterbi (HMM) ==", self.nlg.summarize(vit)]
        path.write_text("\n".join(lines), encoding="utf-8")
        self.set_status(f"บันทึกข้อความแล้ว: {path.relative_to(config.ROOT)}")

    # ================================================================== data collection (feedback loop)
    def selected_label(self):
        return self.class_names[max(0, self.cb_class.current())]

    def refresh_counts(self):
        label = self.selected_label()
        d = config.COLLECTED_DIR / label
        n = len([p for p in d.iterdir() if p.suffix.lower() in config.IMAGE_EXTS]) if d.exists() else 0
        self.lbl_count.configure(text=f"เก็บแล้ว {n} ภาพ")

    def save_sample(self, label):
        from mudra.io_utils import write_image_rgb
        if self.last_frame is None:
            return
        if self.demo:
            self.set_status("โหมดสาธิตไม่เก็บตัวอย่าง (ใช้กับกล้องจริงเท่านั้น)")
            return
        x1, y1, x2, y2 = [int(round(v)) for v in self.roi]
        crop = self.last_frame[max(0, y1):y2, max(0, x1):x2]
        d = config.COLLECTED_DIR / label
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        write_image_rgb(path, crop)
        self.cb_class.current(self.class_names.index(label))
        self.refresh_counts()
        self.set_status(f"เก็บตัวอย่าง '{self.nlg.short(label)}' -> {path.relative_to(config.ROOT)}")

    def start_burst(self):
        self.burst_left, self.burst_label, self.burst_wait = 20, self.selected_label(), 20

    def _handle_burst(self):
        if self.burst_left <= 0:
            return
        if self.burst_wait > 0:              # นับถอยหลังให้ผู้ใช้จัดท่า
            self.burst_wait -= 1
            self.set_status(f"เตรียมทำท่า '{self.nlg.short(self.burst_label)}' ... เริ่มเก็บใน {self.burst_wait} เฟรม")
            return
        if self.ticks % 3 == 0:
            self.save_sample(self.burst_label)
            self.burst_left -= 1
            if self.burst_left == 0:
                self.set_status(f"เก็บครบ 20 ภาพแล้ว — ขยับมือเล็กน้อยระหว่างเก็บจะได้ข้อมูลหลากหลาย "
                                f"เมื่อเก็บครบทุกปาง กด 'ฝึกโมเดลใหม่'")

    # ================================================================== skin calibration
    def start_calibration(self):
        if self.demo:
            self.set_status("ปรับเทียบสีผิวได้เฉพาะกล้องจริง")
            return
        self.calib_until = time.time() + 3.0

    def _calib_box(self, roi):
        x1, y1, x2, y2 = roi
        s = 0.28 * min(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)

    def _handle_calibration(self, frame, roi):
        if self.calib_until is None:
            self.cv.itemconfigure(self.cv_calib, state="hidden")
            return
        b = self._calib_box(roi)
        self.cv.coords(self.cv_calib, *[v * self.s for v in b])
        self.cv.itemconfigure(self.cv_calib, state="normal")
        left = self.calib_until - time.time()
        if left > 0:
            self.set_status(f"ปรับเทียบสีผิว: วางฝ่ามือให้เต็มกรอบเส้นประ ... {left:.1f} วินาที")
            return
        self.calib_until = None
        try:
            self.skin = SkinModel.fit_gaussian(frame[b[1]:b[3], b[0]:b[2]])
            self.skin.save(config.SKIN_FILE)
            if self.recognizer:
                self.recognizer.skin_model = self.skin
            self.set_status("ปรับเทียบสีผิวเรียบร้อย (บันทึกที่ models/skin_model.json)")
        except Exception as e:
            self.set_status(f"ปรับเทียบไม่สำเร็จ: {e}")

    def reset_skin(self):
        self.skin = SkinModel()
        if config.SKIN_FILE.exists():
            config.SKIN_FILE.unlink()
        if self.recognizer:
            self.recognizer.skin_model = self.skin
        self.set_status("ใช้โมเดลสีผิวค่าเริ่มต้น (ช่วง Cr/Cb)")

    # ================================================================== retraining (background thread)
    def start_training(self):
        if self.training:
            return
        self.training = True
        self.btn_train.configure(state="disabled", text="กำลังฝึก...")

        def work():
            from mudra.trainer import run_training
            try:
                meta = run_training(log=lambda s: self.msgs.put(("log", s)))
                self.msgs.put(("done", meta))
            except Exception as e:
                self.msgs.put(("error", repr(e)))

        threading.Thread(target=work, daemon=True).start()

    def _process_messages(self):
        while True:
            try:
                kind, payload = self.msgs.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                text = payload.strip()
                if text:
                    self.set_status("[ฝึกโมเดล] " + text)
            elif kind == "done":
                self.training = False
                self.btn_train.configure(state="normal", text="ฝึกโมเดลใหม่")
                self.load_model()
                self._update_model_label()
                msg = (f"ฝึกโมเดลเสร็จ: {payload['model']}\n"
                       f"Test accuracy: {payload['test']['accuracy'] * 100:.1f}%")
                if payload.get("real_test"):
                    msg += f"\nภาพจริงจากกล้อง (test): {payload['real_test']['accuracy'] * 100:.1f}%"
                self.set_status(msg.replace("\n", "  •  "))
                if not self.max_ticks:
                    messagebox.showinfo("ฝึกโมเดลเสร็จแล้ว", msg)
            elif kind == "error":
                self.training = False
                self.btn_train.configure(state="normal", text="ฝึกโมเดลใหม่")
                self.set_status("ฝึกโมเดลไม่สำเร็จ: " + payload)

    def quit(self):
        try:
            self.cam.release()
        finally:
            self.root.destroy()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ระบบตรวจจับปางพระพุทธรูปจากท่ามือ (กล้อง -> ข้อความ)")
    ap.add_argument("--camera", type=int, default=0, help="หมายเลขกล้อง (ค่าเริ่มต้น 0)")
    ap.add_argument("--demo", action="store_true", help="ไม่ใช้กล้อง: วนภาพจาก Raw Data")
    ap.add_argument("--no-mirror", action="store_true", help="ไม่กลับด้านภาพกล้องแบบกระจก")
    ap.add_argument("--test-ticks", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    from mudra.io_utils import Camera, DemoCamera
    demo = a.demo
    cam = None
    if not demo:
        try:
            cam = Camera(a.camera)
        except Exception as e:
            print(f"เปิดกล้องไม่ได้ ({e}) -> สลับเป็นโหมดสาธิต")
            demo = True
    if demo:
        items = [(s.load(), s.roi) for s in load_sources(include_collected=False)]
        cam = DemoCamera(items)

    root = tk.Tk()
    MudraApp(root, cam, demo=demo, mirror=not a.no_mirror, max_ticks=a.test_ticks or None)
    root.update_idletasks()                   # ให้ Tk คำนวณขนาดหน้าต่างเอง แล้ววางชิดบนกลางจอ
    x = max(0, (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2)
    root.geometry(f"+{x}+0")
    root.mainloop()


if __name__ == "__main__":
    main()
