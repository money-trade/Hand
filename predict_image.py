"""
predict_image.py — ทำนายปางจากไฟล์รูป (ไม่ต้องใช้กล้อง)

ตัวอย่าง:
    python predict_image.py "Raw Data/abhaya.jpg"                       # ใช้ ROI จาก annotations (ถ้ามี)
    python predict_image.py my_hand.jpg --roi 100 150 500 600          # กำหนดกรอบมือเอง
    python predict_image.py my_hand_crop.png                            # ไม่มี ROI = ใช้ทั้งภาพ
"""
import argparse
import json
import sys
from pathlib import Path

from mudra import config
from mudra.dataset import load_mudra_info
from mudra.io_utils import read_image_rgb
from mudra.nlp import ThaiTextGenerator
from mudra.pipeline import MudraRecognizer


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ทำนายปางพระพุทธรูปจากไฟล์รูปท่ามือ")
    ap.add_argument("images", nargs="+", help="ไฟล์รูป")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="ตัวคูณเกณฑ์ยอมรับของด่าน open-set (>1 ผ่อนปรน, <1 เข้มงวด)")
    a = ap.parse_args()

    rec = MudraRecognizer()
    rec.sensitivity = a.sensitivity
    nlg = ThaiTextGenerator(load_mudra_info())
    ann = json.loads(config.ANNOTATION_FILE.read_text(encoding="utf-8"))
    ann = {it["file"]: it for it in ann["images"]}

    for f in a.images:
        p = Path(f)
        img = read_image_rgb(p)
        h, w = img.shape[:2]
        if a.roi:
            roi = tuple(a.roi)
        elif p.name in ann:
            roi = tuple(ann[p.name]["roi"])
            for x1, y1, x2, y2 in ann[p.name].get("exclude", []):
                img[int(y1):int(y2), int(x1):int(x2)] = 127
        else:
            roi = (0, 0, w, h)
        res = rec.analyze(img, roi)
        print(f"\n{p.name}")
        if res["novelty"] is not None:
            print(f"  ระยะห่างจากฐานข้อมูล {res['novelty']:.2f} (เกณฑ์ {rec.sensitivity:.2f})")
        if res["status"] != "known":
            print("  " + nlg.describe(None, 0, state=res["status"], reason=res["reason"])["sentence"])
            continue
        order = res["probs"].argsort()[::-1]
        d = nlg.describe(res["label"], res["confidence"], rec.class_names[order[1]])
        print(f"  {d['sentence']}")
        print(f"  ({d['subtitle']}) ความหมาย: {d['meaning']}")
        for i in order[:3]:
            print(f"    {rec.class_names[i]:16s} {res['probs'][i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
