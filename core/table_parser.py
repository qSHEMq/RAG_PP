# core/table_parser.py
"""Парсинг товарных позиций из grid-таблиц."""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any, Tuple

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
    "nds_rate": [r"став.*ндс", r"ндс.*%", r"%\s*ндс", r"ставка\s*,?\s*%"],
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


def _approx_equal(a: Optional[float], b: Optional[float], rel_tol: float = 1e-3, abs_tol: float = 0.01) -> bool:
    """Проверка равенства чисел с допуском."""
    if a is None or b is None:
        return False
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return False
    diff = abs(a_f - b_f)
    return diff <= max(abs_tol, rel_tol * max(abs(a_f), abs(b_f)))


def _has_letters(s: str) -> bool:
    """Проверка наличия букв в строке."""
    return bool(re.search(r"[A-Za-zА-Яа-я]", s))


def _looks_like_unit(s: Optional[str]) -> bool:
    """Грубая эвристика: строка похожа на обозначение единицы измерения."""
    if not s:
        return False
    text = s.strip()
    if not text:
        return False
    low = text.lower().replace(".", "")
    if low in {"шт", "ед", "кг", "г", "л", "м", "м2", "м3", "уп", "упак", "пак", "т", "час", "ч", "сут", "компл", "усл"}:
        return True
    if len(text) <= 3 and re.match(r"^[A-Za-zА-Яа-я]+$", text):
        return True
    return False


def _strip_leading_row_num(s: str) -> str:
    """Убирает ведущий номер строки из ячейки."""
    if not s:
        return ""
    return re.sub(r"^\s*\d+\s*[.)-]?\s*", "", s).strip()


def _split_row_num_and_name(cell: str) -> Tuple[Optional[int], Optional[str]]:
    """Извлекает номер строки и наименование из одной ячейки."""
    if not cell:
        return None, None
    s = cell.strip()
    if not s:
        return None, None
    m = re.match(r"^\s*(\d{1,4})\s*([A-Za-zА-Яа-я].+)$", s)
    if not m:
        return None, None
    try:
        row_num = int(m.group(1))
    except Exception:
        row_num = None
    name = m.group(2).strip()
    return row_num, name or None


def _has_nds_marker(s: str) -> bool:
    """Проверка наличия маркера НДС с учётом OCR-ошибок (Н/Н, Д/D, С/C)."""
    return re.search(r"[нh][дd][сc]", s) is not None


def _normalize_nds_rate(raw: str) -> Optional[str]:
    """Нормализация ставки НДС до формата '20%' / '10%' / '0%' / 'без НДС'."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    low = s.lower()
    if "без" in low and _has_nds_marker(low):
        return "без НДС"
    if "не облагается" in low and _has_nds_marker(low):
        return "без НДС"
    m = re.search(r"(20|10|0)\s*%$", low)
    if not m:
        m = re.search(r"\b(20|10|0)\s*%", low)
    if m:
        letters = re.findall(r"[a-zа-я]+", low)
        if letters and not all(_has_nds_marker(w) for w in letters):
            return None
        return f"{m.group(1)}%"
    return None


def _split_amount_and_rate(cell: str) -> Tuple[Optional[float], Optional[str]]:
    """Пытается извлечь сумму и ставку НДС из одной ячейки."""
    if not cell:
        return None, None
    s = cell.strip()
    if not s:
        return None, None
    low = s.lower()
    if ("без" in low and _has_nds_marker(low)) or ("не облагается" in low and _has_nds_marker(low)):
        return _parse_float(s), "без НДС"
    m = re.search(r"(20|10|0)\s*%\s*(?:н[дd][сc])?\s*[,.;:]?$", low)
    if m:
        amount_part = s[:m.start()]
        return _parse_float(amount_part), f"{m.group(1)}%"
    return None, None


def _find_nds_rate_in_row(row_cells: List[str], col_mapping: Dict[str, int]) -> Tuple[Optional[str], Optional[int]]:
    """Поиск ставки НДС в строке таблицы с приоритетом NDS-колонок."""
    preferred_fields = ["nds_rate", "nds_amount", "amount", "total_amount"]
    for field in preferred_fields:
        idx = col_mapping.get(field)
        if idx is None or idx >= len(row_cells):
            continue
        norm = _normalize_nds_rate(row_cells[idx])
        if norm:
            return norm, idx
    for idx, cell in enumerate(row_cells):
        norm = _normalize_nds_rate(cell)
        if norm:
            return norm, idx
    # Ищем разнесённое "без" + "НДС" по соседним ячейкам
    for idx in range(len(row_cells) - 1):
        left = row_cells[idx].strip().lower()
        right = row_cells[idx + 1].strip().lower()
        if left == "без" and _has_nds_marker(right):
            return "без НДС", idx
    return None, None


def _find_numeric_right(row_cells: List[str], start_idx: int) -> Tuple[Optional[float], Optional[int]]:
    """Поиск первого числового значения справа от указанного индекса."""
    for idx in range(start_idx + 1, len(row_cells)):
        val = _parse_float(row_cells[idx])
        if val is not None:
            return val, idx
    return None, None


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

        row_num_raw = get_cell("row_num")
        name_raw = get_cell("name")
        unit_code_raw = get_cell("unit_code")
        unit_name_raw = get_cell("unit_name")
        quantity_raw = get_cell("quantity")
        price_raw = get_cell("price")
        amount_raw = get_cell("amount")
        nds_rate_raw = get_cell("nds_rate")
        nds_amount_raw = get_cell("nds_amount")
        total_amount_raw = get_cell("total_amount")

        item.row_num = _parse_int(row_num_raw)
        item.name = name_raw or None
        item.unit_code = unit_code_raw or None
        item.unit_name = unit_name_raw or None
        item.quantity = _parse_float(quantity_raw)
        item.price = _parse_float(price_raw)

        item.amount = _parse_float(amount_raw)
        item.nds_rate = nds_rate_raw or None
        item.nds_amount = _parse_float(nds_amount_raw)
        item.total_amount = _parse_float(total_amount_raw)

        # Поправка склейки "row_num + name" в одной ячейке
        row_num_split, name_split = _split_row_num_and_name(row_num_raw)
        if row_num_split is not None and item.row_num is None:
            item.row_num = row_num_split
        if name_split:
            if not item.name or _looks_like_unit(item.name) or (item.unit_name and item.name == item.unit_name):
                item.name = name_split

        if _has_letters(row_num_raw) and (not item.name or _looks_like_unit(item.name) or (item.unit_name and item.name == item.unit_name)):
            candidate = _strip_leading_row_num(row_num_raw)
            if candidate:
                item.name = candidate

        # Поправка ставок НДС при смещении колонок (ставка попала в соседнюю колонку)
        row_cells = [_normalize_cell(c) for c in row]
        merged_amount = None
        merged_rate = None
        merged_idx = None
        for idx, cell in enumerate(row_cells):
            amt, rate = _split_amount_and_rate(cell)
            if rate and merged_rate is None:
                merged_rate = rate
            if amt is not None and merged_amount is None:
                merged_amount = amt
                merged_idx = idx
            if merged_rate and merged_amount is not None:
                break

        rate_norm, rate_idx = _find_nds_rate_in_row(row_cells, col_mapping)
        rate_value = rate_norm or merged_rate
        rate_pos = rate_idx if rate_norm else merged_idx

        if rate_value:
            if not _normalize_nds_rate(nds_rate_raw):
                item.nds_rate = rate_value
            else:
                item.nds_rate = _normalize_nds_rate(nds_rate_raw) or item.nds_rate

            rate_is_zero = rate_value in ("без НДС", "0%")
            if rate_is_zero:
                item.nds_amount = 0.0
                if item.total_amount is None and item.amount is not None:
                    item.total_amount = item.amount
            else:
                nds_rate_idx = col_mapping.get("nds_rate")
                nds_amount_idx = col_mapping.get("nds_amount")

                # Ставка стоит левее колонки nds_rate -> nds_rate колонка вероятно содержит сумму НДС
                if rate_pos is not None and nds_rate_idx is not None and rate_pos < nds_rate_idx:
                    nds_val = None
                    if nds_rate_idx < len(row_cells):
                        nds_val = _parse_float(row_cells[nds_rate_idx])
                    if nds_val is not None:
                        item.nds_amount = nds_val
                        total_candidate = None
                        if nds_amount_idx is not None and nds_amount_idx < len(row_cells):
                            total_candidate = _parse_float(row_cells[nds_amount_idx])
                        if total_candidate is None:
                            total_candidate, _ = _find_numeric_right(row_cells, nds_rate_idx)
                        if total_candidate is not None and (item.total_amount is None or total_candidate > nds_val):
                            item.total_amount = total_candidate
                else:
                    need_shift = False
                    if _normalize_nds_rate(nds_amount_raw):
                        need_shift = True
                    elif rate_pos is not None and nds_rate_idx is not None and rate_pos > nds_rate_idx:
                        need_shift = True
                    elif rate_pos is not None and nds_rate_idx is None:
                        need_shift = True

                    if need_shift and rate_pos is not None:
                        nds_val, nds_idx = _find_numeric_right(row_cells, rate_pos)
                        if nds_val is not None and (_normalize_nds_rate(nds_amount_raw) or item.nds_amount is None):
                            item.nds_amount = nds_val
                            total_val, _ = _find_numeric_right(row_cells, nds_idx)
                            if total_val is not None:
                                if item.total_amount is None or (item.total_amount <= nds_val):
                                    item.total_amount = total_val

        # Поправка суммы без НДС при склейке "сумма+ставка" в одной ячейке
        if merged_amount is not None:
            if item.amount is None:
                item.amount = merged_amount
            else:
                if item.nds_amount is not None and _approx_equal(item.amount, merged_amount + item.nds_amount):
                    if item.total_amount is None:
                        item.total_amount = item.amount
                    item.amount = merged_amount
                elif item.total_amount is not None and _approx_equal(item.total_amount, merged_amount + (item.nds_amount or 0.0)):
                    item.amount = merged_amount
                elif item.amount > merged_amount and (item.nds_amount is None and item.total_amount is None):
                    item.amount = merged_amount

        # Fallback: восстановить сумму без НДС по итогу и НДС
        if item.nds_amount is not None and item.total_amount is not None:
            derived_amount = item.total_amount - item.nds_amount
            if derived_amount > 0:
                if item.amount is None or _approx_equal(item.amount, item.total_amount):
                    item.amount = derived_amount

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
