OCR + YOLO raw JSON
Проект извлекает “сырой” (raw) OCR-слой из изображения документа и сохраняет единый JSON с:

полным результатом OCR inline (структура PaddleOCR, полигоны и текст);

детекциями layout-блоков (YOLO layout);

детекциями таблиц (YOLO tables) и, при наличии, результатами Table Structure Recognition (TSR) для каждой таблицы.

Этот слой не делает бизнес‑структуризацию (реквизиты, таблицы как объекты домена), а даёт максимально подробное представление документа для следующего этапа (AI‑агент, правила, NLP).

Структура проекта
run_all.py
Главная точка входа. Запускает весь пайплайн: паддинг изображения → OCR → YOLO layout → YOLO tables → TSR → запись единого JSON.

export_raw_json.py
Вспомогательные функции:

xyxy_to_xywh(xyxy): конвертирует bbox из формата 
[
x
1
,
y
1
,
x
2
,
y
2
]
[x1,y1,x2,y2] в словарь {x, y, w, h}.

paddle_result_to_dict(res_obj, tmp_json_path): сохраняет результат PaddleOCR/TSR во временный JSON и загружает его как dict.

yolo_layout.py
Обёртка над Ultralytics YOLO для определения layout-блоков (Page-header, Table, Text, Picture, Page-footer и т.п.).
Возвращает список DetBox с полями label, conf, xyxy. Детекции сортируются сверху‑вниз, слева‑направо.

yolo_tables.py
Обёртка над Ultralytics YOLO для поиска таблиц. Фильтрует только класс "Table" и возвращает список TableDet с conf и xyxy.

table_html_to_grid.py
Утилита для преобразования HTML‑таблицы (из TSR) в двумерный массив grid (List[List[str]]), где каждая вложенная строка — это список ячеек таблицы.

ocr.py
Тестовый скрипт для отдельного запуска PaddleOCR. Сейчас основная логика вынесена в run_all.py, этот файл можно использовать для локальных экспериментов или вовсе не использовать.

Зависимости
Проект использует:

paddleocr — распознавание текста и структуры таблиц;

ultralytics — модели YOLO для layout и таблиц;

opencv-python (cv2) — чтение изображений, кропы, добавление бордюров;

beautifulsoup4 — парсинг HTML таблиц.

Рекомендуется управлять зависимостями через pyproject.toml и uv.lock.

Настройка окружения с uv
Инициализация проекта (если pyproject.toml ещё не создан):

bash
uv init --bare
Добавление зависимостей (если нет requirements.txt):

bash
uv add paddleocr ultralytics opencv-python beautifulsoup4
uv lock
uv sync --locked
Если уже есть requirements.txt, его можно импортировать в pyproject.toml:

bash
uv add -r requirements.txt
uv lock
uv sync --locked
uv.lock фиксирует точные версии пакетов для воспроизводимого окружения.

Настройки в run_all.py
В начале run_all.py нужно указать основные пути:

python
IMAGE_PATH = r"D:\OCR_PP\RAG_PP\data\TORG-12\Screenshot_1.png"
YOLO_TABLES_WEIGHTS = r"D:\OCR_PP\RAG_PP\weights\yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
YOLO_LAYOUT_WEIGHTS = r"D:\OCR_PP\RAG_PP\weights\yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
OUT_ROOT = Path(r"D:\OCR_PP\RAG_PP\output")
Дополнительные параметры:

SAVE_ARTIFACTS — если True, сохраняются кропы таблиц и аннотированные изображения OCR;

PAD_LEFT, PAD_TOP, PAD_RIGHT, PAD_BOTTOM — размер белых бордюров, добавляемых к исходному изображению перед OCR/YOLO (нужно, чтобы избежать “обрезания” текста у краёв).

Запуск
После настройки путей и установки зависимостей достаточно выполнить:

bash
python run_all.py
Скрипт:

Загружает исходное изображение.

Добавляет белый паддинг (по умолчанию слева).

Запускает PaddleOCR и получает ocr_raw (полный результат OCR).

Запускает YOLO layout и получает список layout‑боксов.

Запускает YOLO tables и находит таблицы.

Для каждой таблицы запускает TSR (TableStructureRecognition), получает tsr_raw, опционально HTML и grid.

Формирует единый JSON и сохраняет его в выходную директорию.

Выходные данные
Результаты записываются в:

text
<OUT_ROOT>/<image_stem>/
Где <image_stem> — имя файла изображения без расширения.

Основной файл:

<image_stem>_raw.json — итоговый raw JSON.

Дополнительные артефакты (если SAVE_ARTIFACTS = True):

<image_stem>_annotated/ — изображения с разметкой OCR;

table_crop_<i>.png — кропы каждой найденной таблицы.

Формат <image_stem>_raw.json
Основные ключи:

schema_version
Версия схемы raw JSON (например, "0.1").

source
Информация об источнике (image_path, комментарий о координатах).

image
Размер padded‑изображения и параметры паддинга:

json
"image": {
  "width": <int>,
  "height": <int>,
  "pad": { "left": <int>, "top": <int>, "right": <int>, "bottom": <int> }
}
layout
Массив layout‑детекций:

json
{
  "layout_index": 1,
  "label": "Table",
  "conf": 0.12,
  "bbox_xyxy": [x1, y1, x2, y2],
  "bbox_xywh": {"x": x1, "y": y1, "w": w, "h": h}
}
ocr_raw
Полный результат PaddleOCR (как возвращает библиотека, но встроен inline в JSON).

tables
Массив найденных таблиц:

json
{
  "table_index": 1,
  "yolo_conf": 0.42,
  "bbox_xyxy": [x1, y1, x2, y2],
  "bbox_xywh": {"x": x1, "y": y1, "w": w, "h": h},
  "crop_image": "path/to/table_crop_1.png",
  "tsr_raw": { ... },          // полный raw-результат TSR
  "html": "<table>...</table>",// если извлечён HTML
  "grid": [["cell11", "cell12"], ["cell21", "cell22"]]
}
Переход к следующему слою
Этот raw JSON — база для:

извлечения реквизитов (шапка: дата, номер, контрагент, ИНН, КПП, адрес, валюта, ставка НДС);

структурирования табличной части (строки, колонки, количества, цены, суммы, НДС);

валидаций (сквозная нумерация строк, проверка сумм, согласованность НДС).

Пост‑процессинг (AI‑агент, правила, NLP/NER) будет работать поверх полей ocr_raw, layout и tables, не меняя этот формат JSON.