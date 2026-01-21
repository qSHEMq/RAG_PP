# core/pipeline.py
"""Основной пайплайн обработки документов."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from paddleocr import PaddleOCR, TableStructureRecognition

from yolo_tables import DocLaynetYoloTables
from yolo_layout import YoloLayout
from table_html_to_grid import html_to_grid
from export_raw_json import xyxy_to_xywh, paddle_result_to_dict
from extractor import extract_structured_document, extract_without_llm
from schemas import StructuredDocument


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна."""
    yolo_weights: str = ""
    output_dir: str = "output"
    save_artifacts: bool = False
    pad_left: int = 40
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    yolo_conf: float = 0.10
    yolo_imgsz: int = 1280
    ollama_model: str = "qwen2.5:0.5b-instruct"
    ollama_host: str = "http://127.0.0.1:11434"
    llm_retries: int = 2


class DocumentPipeline:
    """Пайплайн обработки документов."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._ocr = None
        self._tsr = None
        self._layout_model = None
        self._tables_model = None

    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            self._ocr = PaddleOCR(device="cpu", lang="ru")
        return self._ocr

    def _get_tsr(self) -> TableStructureRecognition:
        if self._tsr is None:
            self._tsr = TableStructureRecognition(model_name="SLANet")
        return self._tsr

    def _get_layout_model(self) -> Optional[YoloLayout]:
        if not self.config.yolo_weights:
            return None
        if self._layout_model is None:
            self._layout_model = YoloLayout(
                self.config.yolo_weights,
                conf=self.config.yolo_conf,
                imgsz=self.config.yolo_imgsz
            )
        return self._layout_model

    def _get_tables_model(self) -> Optional[DocLaynetYoloTables]:
        if not self.config.yolo_weights:
            return None
        if self._tables_model is None:
            self._tables_model = DocLaynetYoloTables(
                self.config.yolo_weights,
                conf=self.config.yolo_conf,
                imgsz=self.config.yolo_imgsz
            )
        return self._tables_model

    @staticmethod
    def add_white_border(img: np.ndarray, left: int = 0, top: int = 0,
                         right: int = 0, bottom: int = 0) -> np.ndarray:
        """Добавление белой рамки к изображению."""
        return cv2.copyMakeBorder(
            img, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

    @staticmethod
    def crop_image(img: np.ndarray, xyxy: Tuple[int, int, int, int], pad: int = 6) -> np.ndarray:
        """Вырезание области из изображения."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w - 1, int(x2) + pad)
        y2 = min(h - 1, int(y2) + pad)
        return img[y1:y2, x1:x2].copy()

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """Конвертация PDF в список изображений."""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF не установлен. Установите: pip install pymupdf")

        images = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Увеличиваем масштаб для лучшего качества OCR
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            # Конвертируем в numpy array (BGR для OpenCV)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
        doc.close()
        return images

    def load_image(self, file_path: str) -> List[np.ndarray]:
        """Загрузка изображения или PDF."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.pdf_to_images(file_path)
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"Не удалось загрузить: {file_path}")
            return [img]

    def _extract_html_from_tsr(self, tsr_raw: dict) -> Optional[str]:
        """Извлечение HTML из результата TSR."""
        if not isinstance(tsr_raw, dict):
            return None
        res = tsr_raw.get("res") or tsr_raw
        if not isinstance(res, dict):
            return None
        html = res.get("structure") or res.get("html") or res.get("table_html")
        # structure может быть списком токенов — объединяем в строку
        if isinstance(html, list):
            html = "".join(html)
        return html

    def process_image(self, img: np.ndarray, name: str = "doc") -> dict:
        """
        Обработка одного изображения.

        Args:
            img: Изображение в формате BGR
            name: Имя для сохранения артефактов

        Returns:
            Сырой JSON с результатами OCR и детекции
        """
        cfg = self.config
        out_dir = Path(cfg.output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Добавляем padding
        img_padded = self.add_white_border(
            img, cfg.pad_left, cfg.pad_top, cfg.pad_right, cfg.pad_bottom
        )
        h, w = img_padded.shape[:2]

        # OCR
        ocr = self._get_ocr()
        ocr_results = ocr.predict(img_padded)

        ocr_raw = None
        if ocr_results:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            ocr_raw = paddle_result_to_dict(ocr_results[0], tmp_path)
            tmp_path.unlink(missing_ok=True)

            if cfg.save_artifacts:
                ocr_results[0].save_to_img(str(out_dir / f"{name}_annotated"))

        # Layout detection
        layout_out = []
        layout_model = self._get_layout_model()
        if layout_model:
            layout_dets = layout_model.detect(img_padded)
            for j, d in enumerate(layout_dets, 1):
                layout_out.append({
                    "layout_index": j,
                    "label": d.label,
                    "conf": d.conf,
                    "bbox_xyxy": list(d.xyxy),
                    "bbox_xywh": xyxy_to_xywh(d.xyxy),
                })

        # Table detection
        tables_out = []
        tables_model = self._get_tables_model()
        if tables_model:
            tables = tables_model.detect(img_padded)
            tsr = self._get_tsr()

            for i, t in enumerate(tables, 1):
                crop_img = self.crop_image(img_padded, t.xyxy, pad=10)

                crop_path = out_dir / f"table_crop_{i}.png"
                if cfg.save_artifacts:
                    cv2.imwrite(str(crop_path), crop_img)

                tsr_results = tsr.predict(crop_img)

                tsr_raw = None
                html = None
                grid = []

                if tsr_results:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                    tsr_raw = paddle_result_to_dict(tsr_results[0], tmp_path)
                    tmp_path.unlink(missing_ok=True)

                    html = self._extract_html_from_tsr(tsr_raw)
                    if isinstance(html, str):
                        grid = html_to_grid(html)

                tables_out.append({
                    "table_index": i,
                    "yolo_conf": t.conf,
                    "bbox_xyxy": list(t.xyxy),
                    "bbox_xywh": xyxy_to_xywh(t.xyxy),
                    "crop_image": str(crop_path) if cfg.save_artifacts else None,
                    "tsr_raw": tsr_raw,
                    "html": html,
                    "grid": grid,
                })

        # Формируем результат
        raw_result = {
            "schema_version": "0.2",
            "source": {"name": name},
            "image": {
                "width": w,
                "height": h,
                "pad": {
                    "left": cfg.pad_left,
                    "top": cfg.pad_top,
                    "right": cfg.pad_right,
                    "bottom": cfg.pad_bottom
                },
            },
            "layout": layout_out,
            "ocr_raw": ocr_raw,
            "tables": tables_out,
        }

        return raw_result

    def process_file(self, file_path: str, extract_fields: bool = True) -> List[StructuredDocument]:
        """
        Полная обработка файла (изображение или PDF).

        Args:
            file_path: Путь к файлу
            extract_fields: Извлекать ли поля через LLM

        Returns:
            Список StructuredDocument (по одному на страницу)
        """
        path = Path(file_path)
        name = path.stem
        images = self.load_image(file_path)

        results = []
        for page_idx, img in enumerate(images):
            page_name = f"{name}_p{page_idx + 1}" if len(images) > 1 else name

            # Обработка изображения
            raw = self.process_image(img, page_name)

            # Сохранение raw JSON
            out_dir = Path(self.config.output_dir) / page_name
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_path = out_dir / f"{page_name}_raw.json"
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

            # Извлечение полей
            if extract_fields:
                try:
                    structured = extract_structured_document(
                        raw,
                        model=self.config.ollama_model,
                        host=self.config.ollama_host,
                        retries=self.config.llm_retries,
                    )
                except Exception as e:
                    structured = StructuredDocument()
                    structured.errors.append(str(e))

                # Сохранение structured JSON
                structured_path = out_dir / f"{page_name}_structured.json"
                # Перед сохранением пробуем валидировать по JSON Schema, если он есть
                structured_json = structured.model_dump()
                schema_path = Path(__file__).parent.parent / "schema" / "structured_document.schema.json"
                validation_errors = []
                if schema_path.exists():
                    try:
                        import jsonschema
                        schema = json.loads(schema_path.read_text(encoding="utf-8"))
                        jsonschema.validate(instance=structured_json, schema=schema)
                    except ModuleNotFoundError:
                        # Если jsonschema не установлен — используем pydantic валидацию как fallback
                        try:
                            StructuredDocument.model_validate(structured_json)
                        except Exception as e:
                            validation_errors.append(str(e))
                    except Exception as e:
                        validation_errors.append(str(e))

                if validation_errors:
                    structured.errors.append("Schema validation failed: " + "; ".join(validation_errors))

                structured_path.write_text(
                    structured.model_dump_json(indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                results.append(structured)
            else:
                # Без LLM — используем прямое извлечение из OCR
                try:
                    structured = extract_without_llm(raw)
                except Exception as e:
                    structured = StructuredDocument()
                    structured.errors.append(f"OCR extraction failed: {e}")

                # Сохранение structured JSON
                structured_path = out_dir / f"{page_name}_structured.json"
                structured_path.write_text(
                    structured.model_dump_json(indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                results.append(structured)

        return results

    def process_batch(self, file_paths: List[str], extract_fields: bool = True) -> List[Tuple[str, List[StructuredDocument]]]:
        """
        Пакетная обработка файлов.

        Args:
            file_paths: Список путей к файлам
            extract_fields: Извлекать ли поля через LLM

        Returns:
            Список кортежей (путь, результаты)
        """
        results = []
        for file_path in file_paths:
            try:
                docs = self.process_file(file_path, extract_fields)
                results.append((file_path, docs))
            except Exception as e:
                error_doc = StructuredDocument()
                error_doc.errors.append(f"Failed to process {file_path}: {e}")
                results.append((file_path, [error_doc]))
        return results


def find_yolo_weights() -> Optional[str]:
    """Поиск весов YOLO в типичных местах."""
    possible_paths = [
        Path(__file__).parent.parent / "weights" / "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
        Path.home() / "weights" / "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
        Path("/weights/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"),
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    return None
