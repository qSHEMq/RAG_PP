# core/yolo_tables.py
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

@dataclass
class TableDet:
    conf: float
    xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2

class DocLaynetYoloTables:
    def __init__(self, weights_path: str, conf: float = 0.25, imgsz: int = 1024):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr) -> List[TableDet]:
        r = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None:
            return []

        names = r.names  # {0: 'Caption', ... 8: 'Table', ...}
        h, w = image_bgr.shape[:2]
        dets: List[TableDet] = []

        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            if label != "Table":
                continue

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            dets.append(TableDet(conf=float(b.conf[0].item()), xyxy=(x1, y1, x2, y2)))

        dets.sort(key=lambda d: (d.xyxy[1], d.xyxy[0]))  # сверху-вниз
        return dets
