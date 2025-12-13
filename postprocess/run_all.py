# run_all.py
from __future__ import annotations

from pathlib import Path
import json

import cv2
from paddleocr import PaddleOCR, TableStructureRecognition

from yolo_tables import DocLaynetYoloTables
from yolo_layout import YoloLayout
from table_html_to_grid import html_to_grid
from export_raw_json import xyxy_to_xywh, paddle_result_to_dict

# ==== НАСТРОЙКИ (пути под себя) ====
IMAGE_PATH = r"D:\OCR_PP\RAG_PP\data\TORG-12\Screenshot_1.png"

# ВАЖНО:
# Если у тебя один и тот же вес используется и для layout, и для tables — оставь одинаковым.
# Если разные — укажи разные пути.
YOLO_TABLES_WEIGHTS = r"D:\OCR_PP\RAG_PP\weights\yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
YOLO_LAYOUT_WEIGHTS = r"D:\OCR_PP\RAG_PP\weights\yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"

OUT_ROOT = Path(r"D:\OCR_PP\RAG_PP\output")
# ============================================

# Артефакты: картинки кропов/аннотаций можно оставлять для дебага
SAVE_ARTIFACTS = True

# Анти-обрезание слева (подбери 20..80)
PAD_LEFT = 40
PAD_TOP = 0
PAD_RIGHT = 0
PAD_BOTTOM = 0


def add_white_border(img_bgr, left=0, top=0, right=0, bottom=0):
    return cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


def crop(img, xyxy, pad=6):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w - 1, int(x2) + pad)
    y2 = min(h - 1, int(y2) + pad)
    return img[y1:y2, x1:x2].copy()


def extract_html_from_tsr_dict(tsr_raw: dict | None) -> str | None:
    if not isinstance(tsr_raw, dict):
        return None
    res = tsr_raw.get("res") or tsr_raw
    if not isinstance(res, dict):
        return None
    return res.get("structure") or res.get("html") or res.get("table_html")


def main():
    img_name = Path(IMAGE_PATH).stem
    out_dir = OUT_ROOT / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Load image
    img0 = cv2.imread(IMAGE_PATH)
    if img0 is None:
        raise FileNotFoundError(IMAGE_PATH)

    # 0a) Pad to avoid left-side clipping
    img = add_white_border(img0, left=PAD_LEFT, top=PAD_TOP, right=PAD_RIGHT, bottom=PAD_BOTTOM)
    h, w = img.shape[:2]

    # 1) OCR -> inline dict
    ocr = PaddleOCR(device="cpu", lang="ru")
    ocr_results = ocr.predict(img)  # передаем изображение напрямую

    ocr_raw = None
    if ocr_results:
        tmp_ocr_json = out_dir / "__tmp_ocr.json"
        ocr_raw = paddle_result_to_dict(ocr_results[0], tmp_ocr_json)
        tmp_ocr_json.unlink(missing_ok=True)

        if SAVE_ARTIFACTS:
            # аннотации для отладки
            ocr_results[0].save_to_img(str(out_dir / f"{img_name}_annotated"))

    # 2) YOLO layout -> inline
    layout_model = YoloLayout(YOLO_LAYOUT_WEIGHTS, conf=0.10, imgsz=1280)
    layout_dets = layout_model.detect(img)

    layout_out = []
    for j, d in enumerate(layout_dets, 1):
        layout_out.append({
            "layout_index": j,
            "label": d.label,
            "conf": d.conf,
            "bbox_xyxy": list(d.xyxy),
            "bbox_xywh": xyxy_to_xywh(d.xyxy),
        })

    # 3) YOLO tables -> bbox regions
    tables_model = DocLaynetYoloTables(YOLO_TABLES_WEIGHTS, conf=0.10, imgsz=1280)
    tables = tables_model.detect(img)

    # 4) TSR per table -> inline
    tsr = TableStructureRecognition(model_name="SLANet")
    tables_out = []

    for i, t in enumerate(tables, 1):
        cimg = crop(img, t.xyxy, pad=10)

        crop_path = out_dir / f"table_crop_{i}.png"
        if SAVE_ARTIFACTS:
            cv2.imwrite(str(crop_path), cimg)

        tsr_results = tsr.predict(cimg)  # можно и path, но так проще держать один формат

        tsr_raw = None
        html = None
        grid = []

        if tsr_results:
            tmp_tsr_json = out_dir / f"__tmp_tsr_{i}.json"
            tsr_raw = paddle_result_to_dict(tsr_results[0], tmp_tsr_json)
            tmp_tsr_json.unlink(missing_ok=True)

            html = extract_html_from_tsr_dict(tsr_raw)
            if isinstance(html, str):
                grid = html_to_grid(html)

        tables_out.append({
            "table_index": i,
            "yolo_conf": t.conf,
            "bbox_xyxy": list(t.xyxy),
            "bbox_xywh": xyxy_to_xywh(t.xyxy),
            "crop_image": str(crop_path) if SAVE_ARTIFACTS else None,
            "tsr_raw": tsr_raw,
            "html": html,
            "grid": grid,
        })

    # 5) Unified raw JSON
    out = {
        "schema_version": "0.1",
        "source": {
            "image_path": IMAGE_PATH,
            "note": "All coordinates are in padded-image space. See image.pad to map back."
        },
        "image": {
            "width": w,
            "height": h,
            "pad": {"left": PAD_LEFT, "top": PAD_TOP, "right": PAD_RIGHT, "bottom": PAD_BOTTOM},
        },
        "layout": layout_out,
        "ocr_raw": ocr_raw,
        "tables": tables_out,
    }

    (out_dir / f"{img_name}_raw.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ Done:", out_dir / f"{img_name}_raw.json")


if __name__ == "__main__":
    main()
