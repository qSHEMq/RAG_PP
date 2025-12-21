# export_raw_json.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def xyxy_to_xywh(xyxy: Tuple[int, int, int, int]) -> Dict[str, int]:
    x1, y1, x2, y2 = map(int, xyxy)
    return {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)}


def paddle_result_to_dict(res_obj: Any, tmp_json_path: Path) -> Dict[str, Any]:
    """
    Универсальный способ получить python-dict из PaddleOCR/TableStructureRecognition результата,
    т.к. в твоих примерах гарантированно работает res.save_to_json(...). [file:56]
    """
    res_obj.save_to_json(str(tmp_json_path))
    return json.loads(tmp_json_path.read_text(encoding="utf-8", errors="ignore"))
