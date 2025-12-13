# yolo_layout.py
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

@dataclass
class DetBox:
    label: str
    conf: float
    xyxy: Tuple[int, int, int, int]

class YoloLayout:
    def __init__(self, weights_path: str, conf: float = 0.25, imgsz: int = 1024):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr) -> List[DetBox]:
        r = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets: List[DetBox] = []
        if r.boxes is None:
            return dets

        names = r.names  # class_id -> name
        h, w = image_bgr.shape[:2]

        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            label = str(names.get(cls_id, cls_id))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()  # xyxy доступно у Boxes[web:127]
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            dets.append(DetBox(label=label, conf=conf, xyxy=(x1, y1, x2, y2)))

        # чтение сверху-вниз
        dets.sort(key=lambda d: (d.xyxy[1], d.xyxy[0]))
        return dets
