# core/table_parser.py
"""Парсинг товарных позиций из grid-таблиц."""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any

from schemas import TableItem


# Типичные заголовки колонок в ТОРГ-12 / УПД / Счёт-фактура
COLUMN_PATTERNS = {
    "row_num": [r"№\s*п/?п", r"^№$", r"номер", r"n\s*п/?п"],
    "name": [r"наименован", r"товар", r"описание", r"работ.*услуг", r"грузов"],
    "unit_code": [r"код.*ед", r"океи", r"код$"],
    "unit_name": [r"ед\.?\s*изм", r"единица", r"наим.*ед"],
    "quantity": [r"колич", r"кол-во", r"кол\."],
    "price": [r"цена", r"тариф", r"за\s*ед"],
    "amount": [r"стоим.*без.*ндс", r"сумма.*без.*ндс", r"стоимость"],
    "nds_rate": [r"став.*ндс", r"ндс.*%", r"%\s*ндс"],
    "nds_amount": [r"сумм.*ндс", r"ндс.*руб", r"ндс$"],
    "total_amount": [r"стоим.*с.*ндс", r"сумм.*с.*ндс", r"всего.*с.*ндс", r"итого"],
}


def _normalize_cell(cell: Any) -> str:
    """Нормализация ячейки таблицы."""
    if cell is None:
        return ""
    s = str(cell).strip()
    # Убираем переносы и лишние пробелы
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_float(s: str) -> Optional[float]:
    """Парсинг числа из строки."""
    if not s:
        return None
    # Убираем пробелы, заменяем запятую на точку
    s = s.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    # Убираем валюту и прочий мусор
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in (".", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(s: str) -> Optional[int]:
    """Парсинг целого числа."""
    f = _parse_float(s)
    if f is not None:
        return int(f)
    return None


def _match_column(header: str, patterns: List[str]) -> bool:
    """Проверка соответствия заголовка паттернам."""
    header_lower = header.lower()
    for pat in patterns:
        if re.search(pat, header_lower):
            return True
    return False


def _detect_column_mapping(header_row: List[str]) -> Dict[str, int]:
    """Определение соответствия колонок по заголовкам."""
    mapping = {}
    for col_idx, cell in enumerate(header_row):
        cell_text = _normalize_cell(cell)
        if not cell_text:
            continue
        for field_name, patterns in COLUMN_PATTERNS.items():
            if field_name not in mapping and _match_column(cell_text, patterns):
                mapping[field_name] = col_idx
                break
    return mapping


def _infer_columns_by_data(grid: List[List[Any]]) -> Dict[str, int]:
    """Эвристическое определение колонок анализом содержимого (числовая плотность)."""
    if not grid:
        return {}
    ncols = max(len(r) for r in grid)
    col_scores = [{"numeric": 0, "text": 0} for _ in range(ncols)]
    for row in grid:
        for i in range(ncols):
            cell = _normalize_cell(row[i]) if i < len(row) else ""
            if not cell:
                continue
            if _parse_float(cell) is not None:
                col_scores[i]["numeric"] += 1
            else:
                col_scores[i]["text"] += 1

    # Heuristic: columns with high numeric count are amount/price/quantity
    mapping: Dict[str, int] = {}
    numeric_counts = [(i, col_scores[i]["numeric"]) for i in range(ncols)]
    numeric_counts.sort(key=lambda x: x[1], reverse=True)
    # assign first numeric as amount, second as price, third as quantity (best-effort)
    assigned = 0
    for idx, cnt in numeric_counts:
        if cnt <= 0:
            continue
        if assigned == 0:
            mapping["amount"] = idx
            assigned += 1
        elif assigned == 1:
            mapping["price"] = idx
            assigned += 1
        elif assigned == 2:
            mapping["quantity"] = idx
            assigned += 1
        if assigned >= 3:
            break

    # For name column, pick left-most column with high text count
    text_counts = [(i, col_scores[i]["text"]) for i in range(ncols)]
    text_counts.sort(key=lambda x: x[1], reverse=True)
    for idx, cnt in text_counts:
        if cnt > 0:
            # avoid columns already assigned
            if idx not in mapping.values():
                mapping.setdefault("name", idx)
                break

    return mapping


def _is_header_row(row: List[str]) -> bool:
    """Определение, является ли строка заголовком таблицы."""
    row_text = " ".join(_normalize_cell(c) for c in row).lower()
    # Если есть типичные слова заголовков
    header_keywords = ["наименование", "товар", "описание", "количество", "кол-во", "цена", "стоимость", "ед.", "единиц", "ндс", "№", "итого", "всего"]
    matches = sum(1 for kw in header_keywords if kw in row_text)
    # если встречается 2+ ключевых слов — вероятный заголовок
    if matches >= 2:
        return True
    # иногда заголовок короткий (например только 'наименование' или 'цена'),
    # тогда проверим похожие короткие токены
    tokens = [t for t in re.split(r"\s+", row_text) if t]
    short_hits = sum(1 for t in tokens if any(t.startswith(k[:3]) for k in ("наим", "кол", "цен", "сто", "ед", "ндс", "ит")))
    return short_hits >= 2


def _is_total_row(row: List[str]) -> bool:
    """Определение, является ли строка итоговой."""
    row_text = " ".join(_normalize_cell(c) for c in row).lower()
    total_keywords = ["итого", "всего", "total", "сумма по", "в том числе"]
    return any(kw in row_text for kw in total_keywords)


def _is_data_row(row: List[str], col_mapping: Dict[str, int]) -> bool:
    """Определение, является ли строка строкой данных."""
    if not row:
        return False
    # Должна быть хотя бы одна непустая ячейка
    non_empty = sum(1 for c in row if _normalize_cell(c))
    if non_empty < 2:
        return False
    # Проверяем наличие числовых данных
    for field in ["quantity", "price", "amount"]:
        if field in col_mapping:
            idx = col_mapping[field]
            if idx < len(row):
                val = _parse_float(_normalize_cell(row[idx]))
                if val is not None:
                    return True
    return False


def parse_table_items(grid: List[List[Any]]) -> List[TableItem]:
    """
    Парсинг товарных позиций из grid-таблицы.

    Args:
        grid: 2D массив ячеек таблицы

    Returns:
        Список TableItem с распознанными позициями
    """
    if not grid or len(grid) < 2:
        return []

    items = []
    col_mapping = {}
    header_found = False

    for row_idx, row in enumerate(grid):
        # Ищем строку заголовков
        if not header_found and _is_header_row(row):
            col_mapping = _detect_column_mapping(row)
            # Если заголовок найден, но сопоставление получилось скудным — попробуем объединить с соседней строкой
            if col_mapping and len(col_mapping) < 3 and row_idx + 1 < len(grid):
                next_row = grid[row_idx + 1]
                merged = [f"{_normalize_cell(a)} {_normalize_cell(b)}".strip() for a, b in zip(row + [""] * 100, next_row + [""] * 100)]
                extra_map = _detect_column_mapping(merged)
                for k, v in extra_map.items():
                    if k not in col_mapping:
                        col_mapping[k] = v
            header_found = True
            continue

        # Пропускаем если заголовок ещё не найден
        if not header_found:
            # Пробуем первую строку как заголовок
            col_mapping = _detect_column_mapping(row)
            if col_mapping:
                header_found = True
            else:
                # Попробуем объединить первые 2-3 строки как мультистрочный заголовок
                look_ahead = 3
                for l in range(2, look_ahead + 1):
                    if row_idx + l <= len(grid):
                        max_cols = max(len(r) for r in grid[row_idx:row_idx + l])
                        combo = [
                            " ".join(_normalize_cell(grid[r][c]) if c < len(grid[r]) else "" for r in range(row_idx, row_idx + l)).strip()
                            for c in range(max_cols)
                        ]
                        col_mapping = _detect_column_mapping(combo)
                        if col_mapping:
                            header_found = True
                            break
            # Если всё ещё нет заголовка — попробуем вывести сопоставление по данным таблицы
            if not header_found:
                inferred = _infer_columns_by_data(grid[row_idx:])
                if inferred:
                    col_mapping = inferred
                    header_found = True
            continue

        # Пропускаем итоговые строки
        if _is_total_row(row):
            continue

        # Пропускаем пустые или некорректные строки
        if not _is_data_row(row, col_mapping):
            # Если col_mapping пуст (не найдены заголовки), попробуем определить колонки по данным для всей таблицы
            if not col_mapping:
                inferred = _infer_columns_by_data(grid)
                if inferred:
                    col_mapping = inferred
                else:
                    continue
            else:
                continue

        # Извлекаем данные
        item = TableItem()

        def get_cell(field: str) -> str:
            if field in col_mapping:
                idx = col_mapping[field]
                if idx < len(row):
                    return _normalize_cell(row[idx])
            return ""

        item.row_num = _parse_int(get_cell("row_num"))
        item.name = get_cell("name") or None
        item.unit_code = get_cell("unit_code") or None
        item.unit_name = get_cell("unit_name") or None
        item.quantity = _parse_float(get_cell("quantity"))
        item.price = _parse_float(get_cell("price"))
        item.amount = _parse_float(get_cell("amount"))
        item.nds_rate = get_cell("nds_rate") or None
        item.nds_amount = _parse_float(get_cell("nds_amount"))
        item.total_amount = _parse_float(get_cell("total_amount"))

        # Добавляем только если есть хотя бы наименование, сумма или числовые показатели
        if item.name or item.amount or item.total_amount or item.price or item.quantity:
            # Если имя пустое, заполнить из самой левой текстовой колонки
            if not item.name:
                try:
                    for i in range(len(row)):
                        cell = _normalize_cell(row[i])
                        if cell and _parse_float(cell) is None:
                            item.name = cell
                            break
                except Exception:
                    pass
            items.append(item)

    return items


def extract_totals(grid: List[List[Any]]) -> Dict[str, Optional[float]]:
    """Извлечение итоговых сумм из таблицы."""
    totals = {
        "total_amount": None,
        "total_nds": None,
        "total_with_nds": None,
    }

    for row in grid:
        row_text = " ".join(_normalize_cell(c) for c in row).lower()

        # Ищем итоговую строку
        if "итого" in row_text or "всего" in row_text:
            # Ищем числа в строке
            numbers = []
            for cell in row:
                val = _parse_float(_normalize_cell(cell))
                if val is not None and val > 0:
                    numbers.append(val)

            if numbers:
                # Обычно последнее число - итого с НДС
                if len(numbers) >= 1:
                    totals["total_with_nds"] = numbers[-1]
                if len(numbers) >= 2:
                    totals["total_nds"] = numbers[-2]
                if len(numbers) >= 3:
                    totals["total_amount"] = numbers[-3]

    return totals
