"""
trainer.py — วงจรชีวิต Machine Learning (ML Lifecycle) ครบทุกขั้นตอน

  1. Data Collection        เก็บข้อมูล (Raw Data + ข้อมูลที่เก็บเพิ่มจากกล้อง)
  2. Data Labeling          ติดป้ายกำกับ + กรอบมือ (annotations.json)
  3. EDA                    สำรวจข้อมูล + ภาพตัวอย่างการแบ่งส่วนมือ
  4. Data Cleaning/Preproc  ลบส่วนคอ/คาง, ตัด ROI, ปรับขนาด, แยกสีผิว
  5. Data Augmentation      สร้างข้อมูลเพิ่ม (หมุน ย่อขยาย พลิก แสง สี สัญญาณรบกวน)
  6. Feature Engineering    HOG + Silhouette + Hu moments + Shape
  7. Data Splitting         train / validation / test (stratified)
  8. Feature Scaling + PCA  ปรับสเกล + ลดมิติ (fit กับ train เท่านั้น)
  9. Model Training         kNN, Softmax Regression, MLP
 10. Model Selection        เลือกโมเดลที่ดีที่สุดจาก validation (macro-F1)
 11. Model Evaluation       ประเมินกับ test set + ภาพจริง
 12. Model Deployment       บันทึกโมเดล + เวอร์ชัน -> ใช้งานใน app.py
 13. Monitoring/Feedback    (ใน app.py) เก็บข้อมูลจริงเพิ่ม แล้วกลับไปข้อ 1 เพื่อเทรนใหม่
"""
import json
import shutil
import time
from datetime import datetime

import numpy as np

from . import config
from .dataset import (build_feature_dataset, class_names_from_info, load_sources,
                      make_working, photometric, prepare_source, random_augmentation,
                      render_segmentation_preview, split_dataset)
from .features import extract_features
from .image_ops import draw_rect
from .io_utils import write_image_rgb
from .metrics import evaluate, format_confusion, format_report
from .models import KNNClassifier, NeuralNetClassifier
from .pipeline import save_model
from .preprocessing import PCA, StandardScaler
from .segmentation import SkinModel, extract_hand


def draw_learning_curve(history, path, w=640, h=360):
    """วาดกราฟ loss (น้ำเงิน=train, ส้ม=validation) ด้วย NumPy ล้วน"""
    img = np.full((h, w, 3), 255, np.uint8)
    m = 40
    draw_rect(img, (m, m // 2, w - m // 2, h - m), (180, 180, 180), 1)
    series = [(history.get("train_loss", []), (31, 119, 180)),
              (history.get("val_loss", []), (255, 127, 14))]
    allv = [v for s, _ in series for v in s]
    if not allv:
        return
    vmax, vmin = max(allv), min(0.0, min(allv))
    n = max(len(s) for s, _ in series)
    for yv in np.linspace(vmin, vmax, 5)[1:-1]:
        yy = int(h - m - (yv - vmin) / (vmax - vmin + 1e-9) * (h - 1.5 * m))
        img[yy, m:w - m // 2] = (230, 230, 230)
    for s, color in series:
        if len(s) < 2:
            continue
        xs = m + np.arange(len(s)) / max(1, n - 1) * (w - 1.5 * m)
        ys = h - m - (np.asarray(s) - vmin) / (vmax - vmin + 1e-9) * (h - 1.5 * m)
        tt = np.linspace(0, len(s) - 1, len(s) * 20)
        px = np.interp(tt, np.arange(len(s)), xs).astype(int)
        py = np.interp(tt, np.arange(len(s)), ys).astype(int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                img[np.clip(py + dy, 0, h - 1), np.clip(px + dx, 0, w - 1)] = color
    write_image_rgb(path, img)


def run_training(aug_raw=config.AUG_PER_RAW_IMAGE, aug_collected=config.AUG_PER_COLLECTED,
                 epochs=config.MLP_EPOCHS, seed=config.SEED, include_collected=True, log=print):
    t_start = time.time()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# รายงานการเทรนโมเดล (Training Report)",
              f"\nวันที่: {datetime.now():%Y-%m-%d %H:%M:%S}\n"]

    def step(title):
        log(f"\n=== {title} ===")
        report.append(f"\n## {title}\n")

    # ------------------------------------------------------------------ 1-2
    step("1-2. Data Collection & Labeling")
    class_names = class_names_from_info()
    sources = load_sources(include_collected=include_collected)
    if not sources:
        raise RuntimeError("ไม่พบข้อมูลในโฟลเดอร์ Raw Data")
    rows = ["| คลาส (label) | ภาพดิบ | ภาพจากกล้อง |", "|---|---:|---:|"]
    for c in class_names:
        nr = sum(1 for s in sources if s.label == c and s.kind == "raw")
        nc = sum(1 for s in sources if s.label == c and s.kind == "collected")
        rows.append(f"| {c} | {nr} | {nc} |")
        log(f"  {c:16s} raw={nr}  collected={nc}")
    report.append("\n".join(rows))

    # ------------------------------------------------------------------ 3-4
    step("3-4. EDA & Data Cleaning / Preprocessing")
    skin = SkinModel()
    raw_sources = [s for s in sources if s.kind == "raw"]
    lines = []
    for s in raw_sources:
        img = s.load()
        _, roi = prepare_source(img, s.roi)
        lines.append(f"- `{s.path.name}` ขนาด {img.shape[1]}x{img.shape[0]} px, ROI {list(s.roi)}, "
                     f"ลบส่วนที่ไม่ใช่มือ {len(s.exclude)} กรอบ")
    report.append("\n".join(lines))
    prev = render_segmentation_preview(raw_sources, skin)
    write_image_rgb(config.REPORT_DIR / "segmentation_preview.png", prev)
    report.append("\nภาพผลการแบ่งส่วนมือ (ภาพ ROI | mask สีผิว | ภาพมือที่ crop | silhouette):\n\n"
                  "![segmentation](segmentation_preview.png)")
    log("  บันทึกภาพ reports/segmentation_preview.png")

    # ------------------------------------------------------------------ 5-6
    step("5-6. Data Augmentation & Feature Engineering")
    t0 = time.time()
    ds = build_feature_dataset(sources, class_names, skin, aug_raw, aug_collected, seed, log)
    X, y = ds["X"], ds["y"]
    log(f"  ได้ข้อมูล {len(X)} ตัวอย่าง, ฟีเจอร์ {X.shape[1]} มิติ ({time.time() - t0:.1f} วินาที)")
    report.append(f"- augment ภาพดิบภาพละ {aug_raw} ครั้ง, ภาพจากกล้องภาพละ {aug_collected} ครั้ง\n"
                  f"- ได้ข้อมูลทั้งหมด **{len(X)}** ตัวอย่าง, เวกเตอร์ฟีเจอร์ **{X.shape[1]}** มิติ "
                  f"(HOG 1764 + Silhouette 256 + Hu 7 + Shape 8)")

    # ------------------------------------------------------------------ 7
    step("7. Data Splitting (train / validation / test)")
    sp = split_dataset(ds, sources, seed=seed)
    for k, v in sp.items():
        log(f"  {k:5s}: {len(v)}")
    report.append(" / ".join(f"{k} = {len(v)}" for k, v in sp.items()))
    Xtr, ytr = X[sp["train"]], y[sp["train"]]
    Xva, yva = X[sp["val"]], y[sp["val"]]
    Xte, yte = X[sp["test"]], y[sp["test"]]

    # ------------------------------------------------------------------ 8
    step("8. Feature Scaling & Dimensionality Reduction (PCA)")
    t0 = time.time()
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(config.PCA_MAX_COMPONENTS, config.PCA_VARIANCE, seed=seed)
    Ztr = pca.fit_transform(scaler.transform(Xtr), log=log)
    Zva = pca.transform(scaler.transform(Xva))
    Zte = pca.transform(scaler.transform(Xte))
    cum = float(pca.explained_variance_ratio_.sum())
    report.append(f"StandardScaler -> PCA: {X.shape[1]} มิติ -> **{Ztr.shape[1]}** มิติ "
                  f"(อธิบายความแปรปรวน {cum * 100:.1f}%) ใช้เวลา {time.time() - t0:.1f} วินาที")

    # ------------------------------------------------------------------ 9
    step("9. Model Training")
    nc = len(class_names)
    candidates = [
        ("knn", KNNClassifier(config.KNN_K, nc)),
        ("nn", NeuralNetClassifier(Ztr.shape[1], nc, hidden=(), lr=3e-3, l2=1e-4, dropout=0.0,
                                   epochs=epochs, patience=config.MLP_PATIENCE, seed=seed)),
        ("nn", NeuralNetClassifier(Ztr.shape[1], nc, hidden=config.MLP_HIDDEN, lr=config.MLP_LR,
                                   l2=config.MLP_L2, dropout=config.MLP_DROPOUT, epochs=epochs,
                                   batch_size=config.MLP_BATCH, patience=config.MLP_PATIENCE,
                                   seed=seed)),
    ]
    results = []
    rows = ["| โมเดล | Train acc | Val acc | Val macro-F1 | เวลาเทรน (วินาที) |", "|---|---:|---:|---:|---:|"]
    for mtype, model in candidates:
        log(f"  >> {model.name}")
        t0 = time.time()
        model.fit(Ztr, ytr, Zva, yva, log=log)
        dt = time.time() - t0
        tr = evaluate(ytr, model.predict(Ztr), nc)
        va = evaluate(yva, model.predict(Zva), nc)
        results.append((mtype, model, va, dt))
        log(f"     train acc {tr['accuracy'] * 100:.2f}%  val acc {va['accuracy'] * 100:.2f}%  "
            f"val F1 {va['macro_f1']:.4f}  ({dt:.1f}s)")
        rows.append(f"| {model.name} | {tr['accuracy'] * 100:.2f}% | {va['accuracy'] * 100:.2f}% | "
                    f"{va['macro_f1']:.4f} | {dt:.1f} |")
    report.append("\n".join(rows))

    # ------------------------------------------------------------------ 10
    step("10. Model Selection")
    # เลือก macro-F1 สูงสุด ถ้าเท่ากันเลือกโมเดลที่มาทีหลัง (MLP ให้ค่าความน่าจะเป็นที่เรียบกว่า)
    best_i = max(range(len(results)), key=lambda i: (round(results[i][2]["macro_f1"], 4), i))
    mtype, best, best_va, _ = results[best_i]
    log(f"  เลือก: {best.name} (val macro-F1 = {best_va['macro_f1']:.4f})")
    report.append(f"เลือกโมเดล **{best.name}** (validation macro-F1 สูงสุด = {best_va['macro_f1']:.4f})")
    if getattr(best, "history", None) and best.history.get("val_loss"):
        draw_learning_curve(best.history, config.REPORT_DIR / "learning_curve.png")
        report.append("\nLearning curve (น้ำเงิน = train loss, ส้ม = validation loss):\n\n"
                      "![learning curve](learning_curve.png)")

    # ------------------------------------------------------------------ 11
    step("11. Model Evaluation (Test set)")
    te = evaluate(yte, best.predict(Zte), nc)
    log(f"  test acc {te['accuracy'] * 100:.2f}%  macro-F1 {te['macro_f1']:.4f}")
    report.append("### Test set (ข้อมูลที่โมเดลไม่เคยเห็นตอนเทรน)\n\n" + format_report(te, class_names))
    report.append("\n### Confusion matrix (test)\n\n" + format_confusion(np.array(te["confusion_matrix"]), class_names))

    # ภาพจริง (ไม่ผ่าน augmentation)
    orig = np.nonzero(ds["is_orig"])[0]
    Zo = pca.transform(scaler.transform(X[orig]))
    po = best.predict_proba(Zo)
    lines = ["| ภาพ | ป้ายจริง | ทำนาย | ความมั่นใจ |", "|---|---|---|---:|"]
    n_ok = 0
    for j, i in enumerate(orig):
        s = sources[ds["src"][i]]
        pred = class_names[int(po[j].argmax())]
        n_ok += pred == s.label
        lines.append(f"| {s.path.name} | {s.label} | {pred} | {po[j].max() * 100:.1f}% |")
    log(f"  ภาพจริงทั้งหมด (ไม่ augment): ถูก {n_ok}/{len(orig)}")
    report.append(f"\n### ตรวจสอบกับภาพจริงทั้งหมดที่ไม่ผ่าน augmentation ({n_ok}/{len(orig)} ถูก)\n\n"
                  + "\n".join(lines[:42]) + ("\n| ... | | | |" if len(lines) > 42 else ""))

    col_test = [i for i in sp["test"] if sources[ds["src"][i]].kind == "collected" and ds["is_orig"][i]]
    real = None
    if col_test:
        real = evaluate(y[col_test], best.predict(pca.transform(scaler.transform(X[col_test]))), nc)
        log(f"  ภาพจากกล้อง (test): acc {real['accuracy'] * 100:.2f}%")
        report.append("\n### ภาพจากกล้องจริง (test, แบ่งระดับภาพต้นทาง)\n\n" + format_report(real, class_names))

    # Robustness test: บิดภาพแรงกว่าตอนเทรน (out-of-distribution) ด้วย seed ใหม่
    rob_rows = ["| ระดับการบิดภาพ | ความหมาย | Accuracy | ความมั่นใจเฉลี่ย |", "|---|---|---:|---:|"]
    robustness = {}
    for strength, desc in [(1.0, "เท่าตอนเทรน"), (1.5, "แรงกว่า 1.5 เท่า (หมุน ±27°)"),
                           (2.0, "แรงกว่า 2 เท่า (หมุน ±36°)")]:
        rng = np.random.default_rng(seed + 1000)
        yt, yp, cf = [], [], []
        for s in raw_sources:
            small, roi = prepare_source(s.load(), s.roi)
            for _ in range(30):
                aug = random_augmentation(rng, strength)
                hand = extract_hand(photometric(make_working(small, roi, aug), aug, rng), skin)
                if hand.found:
                    p = best.predict_proba(pca.transform(scaler.transform(extract_features(hand)[None])))[0]
                    yt.append(class_names.index(s.label))
                    yp.append(int(p.argmax()))
                    cf.append(float(p.max()))
        acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else 0.0
        robustness[str(strength)] = acc
        log(f"  robustness x{strength}: acc {acc * 100:.1f}%  (mean conf {np.mean(cf):.2f})")
        rob_rows.append(f"| x{strength} | {desc} | {acc * 100:.1f}% | {np.mean(cf) * 100:.0f}% |")
    report.append("\n### Robustness test (บิดภาพแรงกว่าตอนเทรน ด้วย seed ใหม่)\n\n" + "\n".join(rob_rows))

    report.append("\n> **ข้อควรระวัง:** Raw Data มีเพียงปางละ 1 ภาพ ชุด test ที่มาจากการ augment ภาพเดียวกัน "
                  "จึงวัดได้แค่ความทนทานต่อการหมุน/ย่อ/แสง ไม่ใช่ความแม่นยำกับคนอื่นหรือฉากอื่น "
                  "ควรเก็บข้อมูลจริงจากกล้องเพิ่ม (ปุ่ม \"เก็บตัวอย่าง\" ใน app.py) แล้วเทรนใหม่")

    # ------------------------------------------------------------------ 12
    step("12. Model Deployment (บันทึกโมเดล)")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(config.MODEL_FILE, class_names, scaler, pca, best, mtype)
    config.MODEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.MODEL_FILE, config.MODEL_HISTORY_DIR / f"mudra_model_{stamp}.npz")
    meta = {
        "version": stamp, "model": best.name, "model_type": mtype, "classes": class_names,
        "n_samples": int(len(X)), "feature_dim": int(X.shape[1]), "pca_dim": int(Ztr.shape[1]),
        "split": {k: int(len(v)) for k, v in sp.items()},
        "val": {k: best_va[k] for k in ("accuracy", "macro_f1")},
        "test": {k: te[k] for k in ("accuracy", "macro_f1")},
        "real_test": None if real is None else {k: real[k] for k in ("accuracy", "macro_f1")},
        "robustness": robustness,
        "candidates": [{"model": m.name, "val_acc": va["accuracy"], "val_f1": va["macro_f1"],
                        "train_seconds": dt} for _, m, va, dt in results],
        "train_seconds": time.time() - t_start,
    }
    config.MODEL_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        shown = config.MODEL_FILE.relative_to(config.ROOT)
    except ValueError:
        shown = config.MODEL_FILE
    log(f"  บันทึก {shown} (เวอร์ชัน {stamp})")
    report.append(f"- ไฟล์โมเดล: `models/mudra_model.npz` (สำเนาเวอร์ชัน `models/history/mudra_model_{stamp}.npz`)\n"
                  f"- เวลาเทรนรวม {meta['train_seconds']:.1f} วินาที")

    (config.REPORT_DIR / "training_report.md").write_text("\n".join(report), encoding="utf-8")
    log(f"\nเสร็จสิ้น ({meta['train_seconds']:.1f} วินาที) — รายงาน: reports/training_report.md")
    return meta
