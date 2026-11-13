"""
nlp.py — ส่วน NLP: แปลง "ลำดับผลทำนายรายเฟรม" ให้เป็น "ข้อความภาษาไทย"

หลักการ (เทียบกับงานรู้จำเสียงพูด / Sequence labeling ใน NLP)
  1. Hidden Markov Model (HMM)
       - สถานะซ่อน  = ปางที่ผู้ใช้กำลังทำจริง (+ สถานะ "ไม่มีมือ")
       - การสังเกต  = ความน่าจะเป็นจากโมเดลจำแนกภาพในแต่ละเฟรม (มีสัญญาณรบกวน)
       - Forward algorithm  -> กรองผลแบบเวลาจริง (ลดอาการกระพริบ)
       - Viterbi algorithm  -> ถอดรหัสลำดับที่เป็นไปได้มากที่สุดทั้งช่วง
  2. Tokenization แบบ CTC — ยุบโทเคนที่ซ้ำติดกัน และตัดโทเคนว่าง ("ไม่มีมือ")
  3. Natural Language Generation (NLG) แบบใช้แม่แบบ (template-based)
       - เลือกถ้อยคำตามระดับความมั่นใจ (มั่นใจ / น่าจะ / ไม่แน่ใจ)
       - สรุปลำดับท่าเป็นประโยค
"""
import time

import numpy as np

from . import config

NONE_TOKEN = "<none>"


# ======================================================================
# 1) Hidden Markov Model
# ======================================================================
def sticky_transition_matrix(n_states, stay_prob):
    """เมทริกซ์การเปลี่ยนสถานะ: มีแนวโน้มอยู่สถานะเดิม (ท่ามือไม่เปลี่ยนทุกเฟรม)"""
    A = np.full((n_states, n_states), (1 - stay_prob) / (n_states - 1))
    np.fill_diagonal(A, stay_prob)
    return A


class HMMFilter:
    """Forward algorithm แบบออนไลน์: alpha_t ∝ emission_t * (alpha_{t-1} · A)"""

    def __init__(self, n_classes, stay_prob=config.HMM_STAY_PROB):
        self.n = n_classes + 1                          # + สถานะ "ไม่มีมือ" (ตัวสุดท้าย)
        self.A = sticky_transition_matrix(self.n, stay_prob)
        self.reset()

    def reset(self):
        self.alpha = np.full(self.n, 1.0 / self.n)

    def emission(self, class_probs):
        """แปลงผลจากโมเดลภาพเป็นความน่าจะเป็นของการสังเกต (รวมสถานะไม่มีมือ)"""
        e = np.full(self.n, 1e-3)
        if class_probs is None:
            e[-1] = 1.0
        else:
            e[:-1] = np.asarray(class_probs) * 0.98 + 1e-3
            e[-1] = 0.02
        return e

    def update(self, class_probs):
        pred = self.alpha @ self.A
        a = pred * self.emission(class_probs)
        self.alpha = a / a.sum()
        return self.alpha


def viterbi(emissions, A, prior=None):
    """
    Viterbi decoding (log-space): หาลำดับสถานะที่มีความน่าจะเป็นสูงสุด
    emissions: (T, N)   A: (N, N)
    """
    emissions = np.asarray(emissions, np.float64)
    T, N = emissions.shape
    logA = np.log(A + 1e-12)
    logE = np.log(emissions + 1e-12)
    prior = np.full(N, 1.0 / N) if prior is None else prior
    delta = np.log(prior + 1e-12) + logE[0]
    back = np.zeros((T, N), np.int64)
    for t in range(1, T):
        scores = delta[:, None] + logA          # (from, to)
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + logE[t]
    path = np.zeros(T, np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ======================================================================
# 2) Tokenization (CTC-style collapse)
# ======================================================================
def collapse_tokens(labels, blank=NONE_TOKEN, min_run=1):
    """
    ยุบลำดับรายเฟรมให้เป็นลำดับโทเคน:
      [a a a <none> <none> b b a a] -> [a b a]
    ช่วงที่สั้นกว่า min_run เฟรมถือเป็นสัญญาณรบกวน
    """
    out, prev, run = [], None, 0
    runs = []
    for lab in list(labels) + [None]:
        if lab == prev:
            run += 1
            continue
        if prev is not None:
            runs.append((prev, run))
        prev, run = lab, 1
    for lab, r in runs:
        if lab == blank or r < min_run:
            continue
        if not out or out[-1] != lab:
            out.append(lab)
    return out


class TranscriptBuilder:
    """
    สร้างบทถอดความแบบเวลาจริง: เพิ่มโทเคนเมื่อท่า "นิ่ง" และทายตรงกันต่อเนื่อง STABLE_FRAMES เฟรม
    และบันทึกลำดับ emission ทั้งหมดไว้สำหรับถอดรหัสด้วย Viterbi ภายหลัง
    """

    def __init__(self, class_names, stable_frames=config.STABLE_FRAMES,
                 conf_threshold=config.CONF_THRESHOLD):
        self.class_names = list(class_names)
        self.stable_frames = stable_frames
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self):
        self.tokens = []             # [(label, confidence, timestamp)]
        self.candidate, self.count = None, 0
        self.last_label = None
        self.emissions = []

    def update(self, state_label, confidence, is_still, emission=None):
        """คืนค่าโทเคนใหม่ (ถ้าเพิ่งยืนยันได้) ไม่เช่นนั้นคืน None"""
        if emission is not None:
            self.emissions.append(np.asarray(emission, np.float32))
            if len(self.emissions) > 5000:
                self.emissions = self.emissions[-5000:]
        if state_label == NONE_TOKEN:
            self.candidate, self.count = None, 0
            self.last_label = None             # เอามือออก = ตัวคั่น (ทำท่าเดิมซ้ำได้)
            return None
        if confidence < self.conf_threshold or not is_still:
            self.count = max(0, self.count - 1)
            return None
        if state_label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = state_label, 1
        if self.count >= self.stable_frames and state_label != self.last_label:
            self.last_label = state_label
            tok = (state_label, float(confidence), time.time())
            self.tokens.append(tok)
            return tok
        return None

    def viterbi_tokens(self, A):
        """ถอดรหัสบันทึกทั้งหมดด้วย Viterbi แล้วยุบเป็นโทเคน (ใช้ตอนสรุป/บันทึกไฟล์)"""
        if not self.emissions:
            return []
        path = viterbi(np.stack(self.emissions), A)
        names = self.class_names + [NONE_TOKEN]
        return collapse_tokens([names[i] for i in path], min_run=self.stable_frames // 2)


# ======================================================================
# 3) Natural Language Generation (ภาษาไทย)
# ======================================================================
class ThaiTextGenerator:
    """สร้างข้อความภาษาไทยจากผลทำนาย โดยใช้คลังความรู้ mudra_info.json"""

    def __init__(self, info):
        self.info = info["mudras"]

    def name(self, label):
        return self.info.get(label, {}).get("name_th", label)

    def short(self, label):
        return self.info.get(label, {}).get("short_th", label)

    def describe(self, label, confidence, runner_up=None, state="ok"):
        """
        คืน dict ข้อความสำหรับแสดงผล
          state: "no_model" | "no_hand" | "moving" | "ok"
        """
        if state == "no_model":
            return {"title": "ยังไม่มีโมเดล", "subtitle": "",
                    "sentence": "กรุณาฝึกโมเดลก่อน: python train.py", "hand": "", "meaning": ""}
        if state == "no_hand":
            return {"title": "ไม่พบมือ", "subtitle": "",
                    "sentence": "กรุณายกมือให้อยู่ในกรอบสี่เหลี่ยมบนภาพกล้อง", "hand": "", "meaning": ""}
        d = self.info.get(label, {})
        pct = confidence * 100
        title = d.get("name_th", label)
        if state == "moving":
            sentence = f"กำลังเคลื่อนไหว... ท่าล่าสุดคล้าย{title} ({pct:.0f}%) — ค้างท่าไว้นิ่ง ๆ"
        elif confidence >= 0.85:
            sentence = f"ท่ามือนี้คือ {title} (มั่นใจ {pct:.0f}%)"
        elif confidence >= config.CONF_THRESHOLD:
            sentence = f"ท่ามือนี้น่าจะเป็น {title} (ความมั่นใจ {pct:.0f}%)"
        else:
            alt = f" หรือ {self.name(runner_up)}" if runner_up and runner_up != label else ""
            sentence = f"ยังไม่แน่ใจ อาจเป็น {title}{alt} — ลองจัดท่าให้ชัดขึ้น"
            title = "ไม่แน่ใจ"
        return {"title": title, "subtitle": d.get("name_en", ""), "sentence": sentence,
                "hand": d.get("hand", ""), "meaning": d.get("meaning", "")}

    def token_line(self, token):
        label, conf, ts = token
        return f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] {self.name(label)} ({conf * 100:.0f}%)"

    def summarize(self, labels):
        """สรุปลำดับท่าทั้งหมดเป็นประโยคภาษาไทย"""
        labels = [l for l in labels if l != NONE_TOKEN]
        if not labels:
            return "ยังไม่มีท่าที่ตรวจพบ"
        seq = " → ".join(self.short(l) for l in labels)
        uniq = len(set(labels))
        if len(labels) == 1:
            return f"ตรวจพบ 1 ท่า คือ {self.name(labels[0])}"
        return f"ตรวจพบทั้งหมด {len(labels)} ท่า ({uniq} ปางที่ไม่ซ้ำกัน) ตามลำดับ: {seq}"
