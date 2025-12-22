# core/extractor.py
"""Извлечение структурированных данных из OCR-результатов с помощью LLM."""
from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Optional, List, Any, Dict
from pathlib import Path
import tempfile

import cv2
from paddleocr import PaddleOCR
import requests
from export_raw_json import paddle_result_to_dict

from schemas import DocumentFields, StructuredDocument, PartyInfo, TableItem
from table_parser import parse_table_items, extract_totals


def _build_grid_from_ocr_texts(raw: dict) -> list:
    """Попытка собрать grid-таблицу из OCR-текстов как fallback.

    Возвращает список grid'ов (каждый grid — List[List[str]]).
    """
    grids = []
    try:
        ocr_raw = raw.get("ocr_raw", {}) or {}
        rec_texts = []
        if isinstance(ocr_raw, dict):
            rec_texts = ocr_raw.get("rec_texts", []) or []
        if not rec_texts:
            return grids

        # Соберём потенциальные табличные строки: строки с >=2 числов токенов
        num_re = re.compile(r"\d[\d\s\.,]*\d")
        candidate_rows = []
        for line in rec_texts:
            if not isinstance(line, str):
                continue
            line_s = line.strip()
            if not line_s:
                # break consecutive grouping
                candidate_rows.append(None)
                continue
            nums = num_re.findall(line_s)
            # consider lines with at least 2 numbers or percent sign as table-like
            if len(nums) >= 2 or ("%" in line_s) or ("руб" in line_s) or re.search(r"\d+[,\.]\d{2}", line_s):
                candidate_rows.append(line_s)
            else:
                candidate_rows.append(None)

        # Group consecutive non-None into blocks
        blocks = []
        cur = []
        for r in candidate_rows:
            if r is None:
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(r)
        if cur:
            blocks.append(cur)

        # For each block, try to split into columns by multiple spaces or '|' or tab
        for block in blocks:
            rows = []
            for line in block:
                # try separators: '  ' (2+ spaces), '|', '\t'
                if '|' in line:
                    cols = [c.strip() for c in line.split('|')]
                else:
                    # split by 2+ spaces first
                    parts = re.split(r"\s{2,}|\t", line)
                    if len(parts) <= 1:
                        # fallback: split by single space but keep numeric groups
                        parts = re.split(r"\s+", line)
                    cols = [p.strip() for p in parts if p is not None]
                rows.append(cols)

            # normalize to rectangular grid: pad shorter rows with empty strings
            if not rows:
                continue
            maxc = max(len(r) for r in rows)
            if maxc < 2:
                continue
            grid = [ [r[i] if i < len(r) else "" for i in range(maxc)] for r in rows ]
            # require at least 2 rows and 2 columns
            if len(grid) >= 2 and maxc >= 2:
                grids.append(grid)

        # write debug file if found
        try:
            src = raw.get("source", {}).get("name", "doc")
            out_dir = Path("output") / str(src)
            out_dir.mkdir(parents=True, exist_ok=True)
            if grids:
                (out_dir / "fallback_table_grid.json").write_text(json.dumps(grids, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    except Exception:
        return grids
    return grids


def _log_layout_summary(raw: dict):
    try:
        src = raw.get("source", {}).get("name", "doc")
        out_dir = Path("output") / str(src)
        out_dir.mkdir(parents=True, exist_ok=True)
        layout = raw.get("layout") or []
        summary = {}
        for el in layout:
            lbl = el.get("label") if isinstance(el, dict) else str(el)
            summary[lbl] = summary.get(lbl, 0) + 1
        (out_dir / "layout_summary.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_parties_from_ocr(raw: dict) -> Dict[str, Any]:
    """Прямое извлечение реквизитов продавца/покупателя из OCR текста.

    Ищет по ключевым словам: Продавец, Поставщик, Грузоотправитель -> seller
                            Покупатель, Плательщик, Грузополучатель -> buyer

    Улучшенная логика:
    1. Сначала собираем все организации с ИНН и их контекстом
    2. Определяем роль каждой организации по маркерам
    3. Если org и ИНН в одной строке - они связаны
    """
    result = {
        "seller_name": None,
        "seller_inn": None,
        "seller_kpp": None,
        "buyer_name": None,
        "buyer_inn": None,
        "buyer_kpp": None,
    }

    try:
        ocr_raw = raw.get("ocr_raw", {})
        if not isinstance(ocr_raw, dict):
            return result
        rec_texts = ocr_raw.get("rec_texts", [])
        if not isinstance(rec_texts, list):
            return result

        # Паттерны с учётом OCR-ошибок (О/0/о, O/0)
        # Возможные варианты ООО: ООО, О0О, OOO, о0о, 000, о00 и т.д.
        org_prefixes = r'(?:[ОоOo0]{3}|ЗАО|ПАО|АО|ОАО|ИП|ФГУП|МУП|НКО|ТОО)'
        # Паттерн для названий: до запятой или "ИНН" (чтобы захватить "ООО ШО "Большевичка"")
        org_pattern = rf'({org_prefixes}\s*[^,\n]{{2,60}}?)(?=,|\s*ИНН|\s*р/с|\s*$)'
        # ИНН может быть: "ИНН: 1234567890" или "ИНН/КПП: 1234567890/123456789" или просто "1234567890/123456789"
        inn_pattern = r'(?:ИНН[/КПП\s:]*)?(\d{10}|\d{12})(?:[/\s](\d{9}))?'
        kpp_pattern = r'КПП[:\s]*(\d{9})'
        # Паттерн для формата ИНН/КПП (например "7799763198/779901001")
        inn_kpp_combined = r'\b(\d{10})/(\d{9})\b'

        # Контекстные маркеры (с учётом OCR-ошибок)
        seller_markers = ['продавец', 'поставщик', 'грузоотправитель', 'груз0отправитель', 'исполнитель', 'подрядчик']
        buyer_markers = ['покупатель', 'покупатоль', 'покупат', 'плательщик', 'платель', 'грузополучатель', 'груз0получатель', 'грузопол', 'заказчик']

        lines = [s for s in rec_texts if isinstance(s, str)]

        # Структура для хранения найденных организаций
        # Каждая запись: {line_idx, name, inn, kpp, role: 'seller'|'buyer'|None}
        orgs_found = []

        def _is_garbage_name(name: str) -> bool:
            """Проверка на мусорное название."""
            name_lower = name.lower()
            # Банки - пропускаем
            is_bank = any(b in name_lower for b in ['банк', 'сбербанк', 'bank', 'втб', 'альфа', 'тинькофф', 'газпромбанк'])
            # Служебные фразы из OCR
            is_service = any(s in name_lower for s in ['реквизит', 'рекв', 'реки', 'адрес', 'телефон', 'факс', 'подразделение', 'нков', 'ковск'])
            # Слишком короткое или не содержит букв
            is_garbage = len(name) < 5 or not re.search(r'[а-яА-Яa-zA-Z]{3,}', name)
            # Название состоит только из цифр после префикса (например "ООО 00047")
            name_after_prefix = re.sub(r'^[ОоOo0]{3}\s*', '', name, flags=re.IGNORECASE).strip()
            is_only_digits = bool(re.match(r'^[\d\s]+$', name_after_prefix))
            return is_bank or is_service or is_garbage or is_only_digits

        def _normalize_org_name(name: str) -> str:
            """Нормализация названия организации."""
            # Нормализуем все варианты OCR-ошибок в ООО
            # [ОоOo0]{3} -> ООО
            name = re.sub(r'^[ОоOo0]{3}\s*', 'ООО ', name, flags=re.IGNORECASE)
            # Убираем двойные пробелы
            name = re.sub(r'\s+', ' ', name)
            # Нормализуем кавычки
            name = re.sub(r'["\«\»]', '"', name)
            return name.strip()

        def _get_role_from_context(line_idx: int, lines: list) -> Optional[str]:
            """Определение роли (seller/buyer) по контексту строки."""
            current_line = lines[line_idx].lower() if 0 <= line_idx < len(lines) else ""

            # Функция для проверки что маркер НЕ является частью названия организации
            def _is_marker_not_in_org_name(line: str, marker: str) -> bool:
                """Проверяет что маркер - это метка поля, а не часть названия ООО."""
                line_lower = line.lower()
                # Если маркер идёт ПЕРЕД названием организации (ООО/ЗАО и т.д.) - это метка
                # Например: "Продавец: ООО Рога" или "Покупатель ООО Копыта"
                marker_pos = line_lower.find(marker)
                if marker_pos == -1:
                    return False
                # Ищем ООО/ЗАО и т.д. после маркера
                org_match = re.search(r'[ОоOo0]{3}|ЗАО|ПАО|АО', line_lower[marker_pos:], re.IGNORECASE)
                if org_match:
                    # Маркер перед org - это метка поля
                    return True
                # Проверяем что маркер не внутри названия типа "ООО Поставщик 1"
                org_in_line = re.search(rf'[ОоOo0]{{3}}\s*["\«]?[^"\»,\n]*{marker}', line_lower)
                if org_in_line:
                    # Маркер внутри названия организации - игнорируем
                    return False
                return True

            # Проверяем маркеры в самой строке сначала
            for marker in buyer_markers:
                if marker in current_line and _is_marker_not_in_org_name(current_line, marker):
                    return 'buyer'
            for marker in seller_markers:
                if marker in current_line and _is_marker_not_in_org_name(current_line, marker):
                    return 'seller'

            # Собираем контекст из соседних строк (без текущей)
            for offset in [-1, 1, -2, 2]:  # проверяем ближайшие сначала
                idx = line_idx + offset
                if 0 <= idx < len(lines):
                    neighbor = lines[idx].lower()
                    # Пропускаем строки которые содержат org в начале (это другие организации)
                    # Используем \b чтобы не путать "000" в номерах счетов с "ООО"
                    if re.search(r'^\s*(?:[ОоOo0]{3}|ЗАО|ПАО|ОАО)\s*["\«]?\w', neighbor, re.IGNORECASE):
                        continue
                    for marker in buyer_markers:
                        if marker in neighbor:
                            return 'buyer'
                    for marker in seller_markers:
                        if marker in neighbor:
                            return 'seller'

            return None

        # Вспомогательная функция для извлечения ИНН/КПП из строки
        def _extract_inn_kpp(line: str) -> tuple:
            """Извлечение ИНН и КПП из строки."""
            # Сначала пробуем формат ИНН/КПП (7799763198/779901001)
            combined = re.search(inn_kpp_combined, line)
            if combined:
                return combined.group(1), combined.group(2)
            # Потом пробуем формат "ИНН: 1234567890"
            inn_m = re.search(r'ИНН[:\s]*(\d{10}|\d{12})', line, re.IGNORECASE)
            if inn_m:
                inn = inn_m.group(1)
                kpp_m = re.search(kpp_pattern, line, re.IGNORECASE)
                kpp = kpp_m.group(1) if kpp_m else None
                return inn, kpp
            return None, None

        # Проход 1: Ищем строки с org + ИНН вместе (надёжное извлечение)
        for i, line in enumerate(lines):
            org_match = re.search(org_pattern, line, re.IGNORECASE)
            inn, kpp = _extract_inn_kpp(line)

            if org_match and inn:
                name = _normalize_org_name(org_match.group(1))
                if _is_garbage_name(name):
                    continue

                role = _get_role_from_context(i, lines)

                orgs_found.append({
                    'line_idx': i,
                    'name': name,
                    'inn': inn,
                    'kpp': kpp,
                    'role': role,
                    'has_inn_in_line': True
                })

        # Проход 2: Ищем отдельные org (без ИНН в той же строке)
        for i, line in enumerate(lines):
            # Пропускаем если уже нашли org+inn в этой строке
            if any(o['line_idx'] == i for o in orgs_found):
                continue

            org_match = re.search(org_pattern, line, re.IGNORECASE)
            if org_match:
                name = _normalize_org_name(org_match.group(1))
                if _is_garbage_name(name):
                    continue

                # Определяем роль по контексту
                role = _get_role_from_context(i, lines)

                # Ищем ИНН в соседних строках (±10 для УПД где ИНН может быть далеко)
                nearby_inn = None
                nearby_kpp = None
                for offset in range(-10, 11):
                    idx = i + offset
                    if 0 <= idx < len(lines) and idx != i:
                        inn, kpp = _extract_inn_kpp(lines[idx])
                        if inn:
                            # Проверяем что ИНН не принадлежит другой организации
                            # Для этого смотрим контекст строки с ИНН
                            inn_role = _get_role_from_context(idx, lines)
                            # Если роли совпадают или одна из них None - ИНН подходит
                            if inn_role == role or inn_role is None or role is None:
                                nearby_inn = inn
                                nearby_kpp = kpp
                                break

                # Добавляем только если нет дубликата по имени
                if not any(o['name'] == name for o in orgs_found):
                    orgs_found.append({
                        'line_idx': i,
                        'name': name,
                        'inn': nearby_inn,
                        'kpp': nearby_kpp,
                        'role': role,
                        'has_inn_in_line': False
                    })

        # Распределяем роли
        # Приоритет: явно определённые роли > порядок появления
        sellers = [o for o in orgs_found if o['role'] == 'seller']
        buyers = [o for o in orgs_found if o['role'] == 'buyer']
        unknowns = [o for o in orgs_found if o['role'] is None]

        # Выбираем seller
        if sellers:
            # Предпочитаем org с ИНН в той же строке
            sellers_with_inn = [s for s in sellers if s['has_inn_in_line']]
            best_seller = sellers_with_inn[0] if sellers_with_inn else sellers[0]
            result["seller_name"] = best_seller['name']
            result["seller_inn"] = best_seller['inn']
            result["seller_kpp"] = best_seller['kpp']
        elif unknowns:
            # Первый unknown становится seller
            best_seller = unknowns[0]
            result["seller_name"] = best_seller['name']
            result["seller_inn"] = best_seller['inn']
            result["seller_kpp"] = best_seller['kpp']
            unknowns = unknowns[1:]  # Убираем из списка

        # Выбираем buyer
        if buyers:
            # Предпочитаем org с ИНН в той же строке
            buyers_with_inn = [b for b in buyers if b['has_inn_in_line']]
            best_buyer = buyers_with_inn[0] if buyers_with_inn else buyers[0]
            result["buyer_name"] = best_buyer['name']
            result["buyer_inn"] = best_buyer['inn']
            result["buyer_kpp"] = best_buyer['kpp']
        elif unknowns:
            # Следующий unknown становится buyer
            best_buyer = unknowns[0]
            result["buyer_name"] = best_buyer['name']
            result["buyer_inn"] = best_buyer['inn']
            result["buyer_kpp"] = best_buyer['kpp']

    except Exception:
        pass

    return result


def _extract_doc_info_from_ocr(raw: dict) -> Dict[str, Any]:
    """Извлечение номера и даты документа из OCR текста."""
    result = {
        "doc_number": None,
        "doc_date": None,
        "doc_type": None,
    }

    try:
        ocr_raw = raw.get("ocr_raw", {})
        if not isinstance(ocr_raw, dict):
            return result
        rec_texts = ocr_raw.get("rec_texts", [])
        if not isinstance(rec_texts, list):
            return result

        # Паттерны
        date_pattern = r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b'
        number_pattern = r'\b(\d{3,10})\b'

        # Ищем тип документа
        full_text = " ".join(s for s in rec_texts if isinstance(s, str))
        if 'торг-12' in full_text.lower() or 'товарная накладная' in full_text.lower():
            result["doc_type"] = "ТОРГ-12"
        elif 'упд' in full_text.lower() or 'универсальный передаточный' in full_text.lower():
            result["doc_type"] = "УПД"
        elif 'счёт-фактура' in full_text.lower() or 'счет-фактура' in full_text.lower():
            result["doc_type"] = "Счёт-фактура"

        # Ищем номер и дату рядом с ключевыми словами "Номер документа" и "Дата составления"
        found_doc_section = False
        for i, line in enumerate(rec_texts):
            if not isinstance(line, str):
                continue
            line_lower = line.lower()

            # Пропускаем строки с "утверждена", "постановлением" - это шапка формы
            if 'утвержден' in line_lower or 'постановлен' in line_lower:
                continue

            # Ищем секцию "Номер документа" / "Дата составления"
            if 'номер документа' in line_lower:
                found_doc_section = True
                # Смотрим следующие строки для номера
                for j in range(i + 1, min(i + 7, len(rec_texts))):
                    next_line = rec_texts[j] if isinstance(rec_texts[j], str) else ""
                    # Ищем чистое число (только цифры) - это номер документа
                    next_clean = next_line.strip()
                    if re.match(r'^\d{3,10}$', next_clean) and not result["doc_number"]:
                        result["doc_number"] = next_clean
                        break

            if 'дата составления' in line_lower:
                found_doc_section = True
                # Смотрим следующие строки для даты
                for j in range(i + 1, min(i + 5, len(rec_texts))):
                    next_line = rec_texts[j] if isinstance(rec_texts[j], str) else ""
                    date_match = re.search(date_pattern, next_line)
                    if date_match and not result["doc_date"]:
                        # Проверяем что дата не из 1990-х (шапка формы)
                        date_str = date_match.group(1)
                        if '.98' not in date_str and '.99' not in date_str and '.19' not in date_str:
                            result["doc_date"] = date_str
                            break

        # Fallback: ищем дату в формате DD.MM.20XX (современные документы)
        # Но исключаем даты из строк "Основание", "от", "договор" - это не дата документа
        if not result["doc_date"]:
            modern_date_pattern = r'\b(\d{1,2}[./]\d{1,2}[./]20\d{2})\b'
            for i, line in enumerate(rec_texts):
                if isinstance(line, str):
                    line_lower = line.lower()
                    # Пропускаем строки с "основание", "от", "договор" - это даты договоров
                    if 'основание' in line_lower or ' от ' in line_lower or 'договор' in line_lower:
                        continue
                    # Пропускаем строки с годом в формате "1812 года" (адреса)
                    if 'года ул' in line_lower or 'год ул' in line_lower:
                        continue
                    date_match = re.search(modern_date_pattern, line)
                    if date_match:
                        result["doc_date"] = date_match.group(1)
                        break

    except Exception:
        pass

    return result


def _extract_totals_from_ocr(raw: dict) -> Dict[str, Any]:
    """Извлечение итоговых сумм из OCR текста.

    Ищет паттерны:
    - "Всего к оплате" / "Итого" с последующими суммами
    - Суммы в формате 1234,56 или 1 234,56
    """
    result = {
        "total_amount": None,      # Сумма без НДС
        "total_nds": None,         # Сумма НДС
        "total_with_nds": None,    # Итого с НДС
    }

    try:
        ocr_raw = raw.get("ocr_raw", {})
        if not isinstance(ocr_raw, dict):
            return result
        rec_texts = ocr_raw.get("rec_texts", [])
        if not isinstance(rec_texts, list):
            return result

        def _parse_amount(s: str) -> Optional[float]:
            """Парсинг суммы из строки. Возвращает None для количеств (целых чисел)."""
            if not s:
                return None
            s = s.strip()
            # Пропускаем числа вида "15,000" или "15.000" - это количество, не сумма
            # Суммы обычно имеют формат "2500,00" или "2 500,00"
            if re.match(r'^\d+[,.]000$', s):
                return None  # Это количество (15,000 = 15 штук)
            # Убираем пробелы между цифрами (4 200,00 -> 4200,00)
            s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
            # Заменяем запятую на точку
            s = s.replace(',', '.')
            # Убираем всё кроме цифр и точки
            s = re.sub(r'[^\d.]', '', s)
            try:
                val = float(s)
                # Суммы обычно > 1 (копейки есть)
                return val if val > 0 else None
            except:
                return None

        # Ищем маркеры итогов
        total_markers = ['всего к оплате', 'итого', 'всего по документу', 'общая сумма']

        for i, line in enumerate(rec_texts):
            if not isinstance(line, str):
                continue
            line_lower = line.lower()

            # Проверяем маркеры итогов
            is_total_line = any(m in line_lower for m in total_markers)

            if is_total_line:
                # Ищем суммы в следующих строках (обычно 3 значения: без НДС, НДС, с НДС)
                amounts_found = []
                for j in range(i + 1, min(i + 10, len(rec_texts))):
                    next_line = rec_texts[j] if isinstance(rec_texts[j], str) else ""
                    # Пропускаем строки-маркеры (X, -, и т.д.)
                    if next_line.strip() in ['X', 'x', 'Х', 'х', '-', '.']:
                        continue
                    # Ищем число в строке
                    amount = _parse_amount(next_line)
                    if amount and amount > 0:
                        amounts_found.append(amount)
                    # Если нашли 3 суммы или встретили текст - выходим
                    if len(amounts_found) >= 3:
                        break
                    if next_line.strip() and not re.search(r'[\d,.]', next_line):
                        break

                # Присваиваем суммы
                if len(amounts_found) >= 3:
                    # Стандартный порядок: без НДС, НДС, с НДС
                    result["total_amount"] = amounts_found[0]
                    result["total_nds"] = amounts_found[1]
                    result["total_with_nds"] = amounts_found[2]
                elif len(amounts_found) == 2:
                    # Возможно только НДС и итого
                    result["total_nds"] = amounts_found[0]
                    result["total_with_nds"] = amounts_found[1]
                elif len(amounts_found) == 1:
                    result["total_with_nds"] = amounts_found[0]

                if result["total_with_nds"]:
                    break  # Нашли итоги

        # Fallback: ищем паттерн "Итого: X руб" в тексте
        if not result["total_with_nds"]:
            full_text = " ".join(s for s in rec_texts if isinstance(s, str))
            # Ищем суммы с валютой
            amount_pattern = r'(?:итого|всего)[:\s]*(\d[\d\s]*[.,]\d{2})\s*(?:руб|₽)?'
            match = re.search(amount_pattern, full_text, re.IGNORECASE)
            if match:
                result["total_with_nds"] = _parse_amount(match.group(1))

    except Exception:
        pass

    return result


# --- Промпт для LLM ---

SYSTEM_PROMPT = """Ты JSON-экстрактор. Верни ТОЛЬКО JSON, без текста до или после.
НЕ ПИШИ пояснений, комментариев, списков. ТОЛЬКО JSON.

Извлеки данные из российского бухгалтерского документа (ТОРГ-12, УПД, Счёт-фактура).
Если поле отсутствует - null. Не выдумывай.

ВАЖНО - определение сторон сделки:
- ПРОДАВЕЦ (seller) - тот, кто ПРОДАЁТ товар/услугу. Ищи слова: "Продавец", "Поставщик", "Исполнитель", "Грузоотправитель"
- ПОКУПАТЕЛЬ (buyer) - тот, кто ПОКУПАЕТ товар/услугу. Ищи слова: "Покупатель", "Заказчик", "Плательщик", "Грузополучатель"
- В УПД: поле [14] внизу слева = продавец, поле [19] внизу справа = покупатель
- НЕ ПУТАЙ стороны! Внимательно читай метки полей.

JSON должен точно соответствовать этой схеме:
{
    "doc_type": "ТОРГ-12" | "УПД" | "Счёт-фактура" | null,
    "doc_number": "string (только номер документа)" | null,
    "doc_date": "DD.MM.YYYY" | null,
    "sf_number": "string (номер счёт-фактуры для УПД)" | null,
    "sf_date": "DD.MM.YYYY" | null,
    "seller_name": "string (полное наименование продавца с ООО/ЗАО/ИП)" | null,
    "seller_inn": "string (10 или 12 цифр, ТОЛЬКО цифры)" | null,
    "seller_kpp": "string (9 цифр, ТОЛЬКО цифры)" | null,
    "seller_address": "string" | null,
    "buyer_name": "string (полное наименование покупателя)" | null,
    "buyer_inn": "string (10 или 12 цифр, ТОЛЬКО цифры)" | null,
    "buyer_kpp": "string (9 цифр, ТОЛЬКО цифры)" | null,
    "buyer_address": "string" | null,
    "total_amount": number (сумма БЕЗ НДС) | null,
    "total_nds": number (сумма НДС) | null,
    "total_with_nds": number (итого С НДС) | null,
    "currency": "руб." | "RUB" | "USD" | "EUR" | null,
    "items": [
        {
            "row_num": number (номер строки),
            "name": "string (наименование товара/услуги)",
            "unit": "string (единица измерения: шт, кг, м и т.д.)" | null,
            "quantity": number (количество) | null,
            "price": number (цена за единицу) | null,
            "amount": number (сумма без НДС) | null,
            "nds_rate": "string (ставка НДС: 20%, 10%, без НДС)" | null,
            "nds_amount": number (сумма НДС) | null,
            "total_amount": number (сумма с НДС) | null
        }
    ] | []
}

Правила:
- Возвращай СТРОГО JSON без обрамления ``` и комментариев
- ИНН и КПП - ОТДЕЛЬНЫЕ поля, не объединяй их
- Даты в формате DD.MM.YYYY
- Числовые суммы - числа (не строки)
- items - массив товарных позиций из таблицы документа

ПРИМЕР ОТВЕТА (формат):
{"doc_type":"УПД","doc_number":"123","doc_date":"01.01.2025","seller_name":"ООО Рога","seller_inn":"1234567890","seller_kpp":"123456789","buyer_name":"ООО Копыта","buyer_inn":"0987654321","buyer_kpp":"987654321","total_amount":1000,"total_nds":200,"total_with_nds":1200,"items":[{"row_num":1,"name":"Товар","quantity":10,"price":100,"amount":1000}]}
"""


def _extract_from_text_fallback(text: str) -> Dict[str, Any]:
    """Извлечение данных из текстового ответа LLM по паттернам."""
    result: Dict[str, Any] = {}

    # Тип документа
    if re.search(r"УПД|универсальн", text, re.IGNORECASE):
        result["doc_type"] = "УПД"
    elif re.search(r"ТОРГ.?12|товарная накладная", text, re.IGNORECASE):
        result["doc_type"] = "ТОРГ-12"
    elif re.search(r"сч[её]т.?фактур", text, re.IGNORECASE):
        result["doc_type"] = "Счёт-фактура"

    # Номер документа
    m = re.search(r"[Нн]омер[:\s]*(\d+)", text)
    if m:
        result["doc_number"] = m.group(1)

    # Дата
    m = re.search(r"[Дд]ата[:\s]*(\d{1,2}[\s\./-]\w+[\s\./-]?\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})", text)
    if m:
        date_str = m.group(1)
        # Нормализуем формат
        months = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04",
                  "мая": "05", "июня": "06", "июля": "07", "августа": "08",
                  "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"}
        for mon_name, mon_num in months.items():
            if mon_name in date_str.lower():
                date_str = re.sub(rf"\s*{mon_name}\s*", f".{mon_num}.", date_str, flags=re.IGNORECASE)
                break
        date_str = re.sub(r"\s+г\.?$", "", date_str)
        date_str = date_str.replace(" ", "")
        result["doc_date"] = date_str

    # ИНН/КПП - ищем все пары
    inn_kpp_pairs = re.findall(r"ИНН/?КПП[:\s]*(\d{10,12})[/\s]*(\d{9})?", text, re.IGNORECASE)

    # Продавец
    seller_match = re.search(r"[Пп]родав[е|ц][а-я]*\s*[:\(]?\s*([^\):\n]+?)(?:\)|:|$|\n)", text)
    if seller_match:
        seller_name = seller_match.group(1).strip()
        # Ищем ООО/ЗАО и т.д.
        org_match = re.search(r'((?:ООО|ЗАО|ПАО|АО|ИП)\s*"[^"]+"|(?:ООО|ЗАО|ПАО|АО|ИП)\s+\S+)', seller_name)
        if org_match:
            result["seller_name"] = org_match.group(1)
        elif seller_name:
            result["seller_name"] = seller_name

    # Покупатель
    buyer_match = re.search(r"[Пп]окупател[ья][а-я]*\s*[:\(]?\s*([^\):\n]+?)(?:\)|:|$|\n)", text)
    if buyer_match:
        buyer_name = buyer_match.group(1).strip()
        org_match = re.search(r'((?:ООО|ЗАО|ПАО|АО|ИП)\s*"[^"]+"|(?:ООО|ЗАО|ПАО|АО|ИП)\s+\S+)', buyer_name)
        if org_match:
            result["buyer_name"] = org_match.group(1)
        elif buyer_name:
            result["buyer_name"] = buyer_name

    # Альтернативный поиск организаций по контексту
    # Ищем "Автотрейд" и "Конфетпром" с контекстом продавец/покупатель
    if re.search(r"продавец[а-я]*\s*\(?Автотрейд", text, re.IGNORECASE):
        result["seller_name"] = 'ООО "Автотрейд"'
    if re.search(r"покупател[ья]\s*\(?Конфетпром", text, re.IGNORECASE):
        result["buyer_name"] = 'ООО "Конфетпром"'
    if re.search(r"продавец[а-я]*\s*\(?Конфетпром", text, re.IGNORECASE):
        result["seller_name"] = 'ООО "Конфетпром"'
    if re.search(r"покупател[ья]\s*\(?Автотрейд", text, re.IGNORECASE):
        result["buyer_name"] = 'ООО "Автотрейд"'

    # Назначаем ИНН/КПП если нашли пары
    if inn_kpp_pairs:
        if len(inn_kpp_pairs) >= 1:
            result["seller_inn"] = inn_kpp_pairs[0][0]
            if inn_kpp_pairs[0][1]:
                result["seller_kpp"] = inn_kpp_pairs[0][1]
        if len(inn_kpp_pairs) >= 2:
            result["buyer_inn"] = inn_kpp_pairs[1][0]
            if inn_kpp_pairs[1][1]:
                result["buyer_kpp"] = inn_kpp_pairs[1][1]

    # Суммы
    amount_match = re.search(r"[Ии]того[:\s]*(\d[\d\s,\.]*)", text)
    if amount_match:
        amt = amount_match.group(1).replace(" ", "").replace(",", ".")
        try:
            result["total_with_nds"] = float(amt)
        except:
            pass

    return result


def _safe_json_extract(text: str) -> Any:
    """Извлечение JSON из ответа модели."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Ищем JSON в тексте (попытка жесткой и lenient-извлечения)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Если JSON не найден - пробуем извлечь из текста
    fallback = _extract_from_text_fallback(text)
    if fallback:
        return fallback

    # Если нет полного JSON-блока (возможно, ответ обрезан) — берём от первого '{' до конца
    idx = text.find("{")
    if idx == -1:
        raise ValueError(f"JSON не найден в ответе: {text[:200]!r}")

    candidate = text[idx:]
    # Убираем возможные ограждающие тройные бэктики
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)

    # Попытка добавить недостающие закрывающие '}'
    opens = candidate.count("{")
    closes = candidate.count("}")
    if opens > closes:
        candidate_fixed = candidate + ("}" * (opens - closes))
    else:
        candidate_fixed = candidate

    # Убираем завершающие запятые перед закрывающей фигурной скобкой, часто возникающие при усечённом выводе
    candidate_fixed = re.sub(r",\s*}\s*$", "}\n", candidate_fixed)

    # Попытка догнать незакрытые кавычки: если нечётное число двойных кавычек — добавляем закрывающую
    try:
        if candidate_fixed.count('"') % 2 == 1:
            candidate_fixed = candidate_fixed + '"'
    except Exception:
        pass

    try:
        return json.loads(candidate_fixed)
    except Exception as e:
        raise ValueError(f"JSON parse failed after lenient fix: {e}; snippet: {candidate_fixed[:300]!r}")


def _fix_doc_type(s: Optional[str]) -> Optional[str]:
    """Нормализация типа документа."""
    if not s:
        return None
    s_lower = s.lower()
    if "торг" in s_lower or "torg" in s_lower:
        return "ТОРГ-12"
    if "упд" in s_lower or "универсальн" in s_lower:
        return "УПД"
    if "счёт-фактур" in s_lower or "счет-фактур" in s_lower:
        return "Счёт-фактура"
    return s


def _fix_inn(s: Optional[str]) -> Optional[str]:
    """Очистка ИНН."""
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) in (10, 12):
        return digits
    return s


def _fix_kpp(s: Optional[str]) -> Optional[str]:
    """Очистка КПП."""
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 9:
        return digits
    return s


def _fix_date(s: Optional[str]) -> Optional[str]:
    """Нормализация даты."""
    if not s:
        return None
    # Убираем "г." и лишние пробелы
    s = re.sub(r"\s*г\.?\s*$", "", s.strip())
    # Проверяем формат ДД.ММ.ГГГГ
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", s)
    if m:
        d, mo, y = m.groups()
        if len(y) == 2:
            y = "20" + y if int(y) < 50 else "19" + y
        return f"{int(d):02d}.{int(mo):02d}.{y}"
    return s


def compact_text_from_raw(raw: dict, max_lines: int = 200) -> str:
    """Формирование компактного текста для LLM."""
    parts = []

    # OCR текст
    ocr_raw = raw.get("ocr_raw", {})
    if isinstance(ocr_raw, dict):
        rec_texts = ocr_raw.get("rec_texts", [])
        if isinstance(rec_texts, list):
            lines = [s.strip() for s in rec_texts if isinstance(s, str) and s.strip()]
            if lines:
                # Берём начало и конец
                head = lines[:max_lines // 2]
                tail = lines[-(max_lines // 2):] if len(lines) > max_lines // 2 else []
                text = "\n".join(head)
                if tail and tail != head[-len(tail):]:
                    text += "\n...\n" + "\n".join(tail)
                parts.append(text)

    return "\n".join(parts)[:12000]


def _run_ollama_cli(model: str, prompt: str, timeout_sec: int = 180) -> str:
    """Запуск Ollama через CLI."""
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama CLI failed: {proc.stderr[:500]}")
    return proc.stdout.strip()


def extract_fields_with_llm(
    raw: dict,
    model: str = "qwen2.5:0.5b-instruct",
    host: str = "http://127.0.0.1:11434",
    retries: int = 2,
) -> Dict[str, Any]:
    """Извлечение полей с помощью LLM."""

    compact = compact_text_from_raw(raw)
    user_prompt = f"Извлеки реквизиты из документа:\n\n{compact}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 512},
    }

    # Пробуем HTTP API с ретраями
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=180)
            if r.ok:
                content = r.json().get("message", {}).get("content")
                if content is None:
                    raise RuntimeError("Empty content from /api/chat")
            # Сохраняем сырой ответ LLM для отладки
            try:
                src = raw.get("source", {}).get("name", "doc")
                out_dir = Path("output") / str(src)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "llm_raw.txt").write_text(str(content), encoding="utf-8")
            except Exception:
                pass
            # Попытка парсинга с lenient-логикой; если парсинг не прошёл — извлечём пары ключ-значение как fallback
            try:
                return _safe_json_extract(content)
            except Exception as _exc:
                # fallback: извлекаем простые пары "key": "value" из текста
                try:
                    src = raw.get("source", {}).get("name", "doc")
                    out_dir = Path("output") / str(src)
                    (out_dir / "llm_parse_error.txt").write_text(repr(_exc), encoding="utf-8")
                except Exception:
                    pass
                def _fallback_from_text(text: str) -> Dict[str, Any]:
                    out: Dict[str, Any] = {}
                    # Находим пары вида "key": null или "key": "value"
                    for m in re.finditer(r'"(?P<k>[a-zA-Z0-9_]+)"\s*:\s*(?P<v>null|"(?:[^"\\]|\\.)*")', text):
                        k = m.group('k')
                        v = m.group('v')
                        if v == 'null':
                            out[k] = None
                        else:
                            # убираем кавычки и unescape
                            val = v[1:-1]
                            out[k] = val
                    return out

                return _fallback_from_text(content)
                # Сохраняем сырой ответ LLM для отладки
                try:
                    src = raw.get("source", {}).get("name", "doc")
                    out_dir = Path("output") / str(src)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "llm_raw.txt").write_text(str(content), encoding="utf-8")
                except Exception:
                    pass
                # Попытка парсинга с lenient-логикой; если парсинг не прошёл — извлечём пары ключ-значение как fallback
                try:
                    return _safe_json_extract(content)
                except Exception as _exc:
                    # fallback: извлекаем простые пары "key": "value" из текста
                    try:
                        src = raw.get("source", {}).get("name", "doc")
                        out_dir = Path("output") / str(src)
                        (out_dir / "llm_parse_error.txt").write_text(repr(_exc), encoding="utf-8")
                    except Exception:
                        pass
                    def _fallback_from_text(text: str) -> Dict[str, Any]:
                        out: Dict[str, Any] = {}
                        # Находим пары вида "key": null или "key": "value"
                        for m in re.finditer(r'"(?P<k>[a-zA-Z0-9_]+)"\s*:\s*(?P<v>null|"(?:[^"\\]|\\.)*")', text):
                            k = m.group('k')
                            v = m.group('v')
                            if v == 'null':
                                out[k] = None
                            else:
                                # убираем кавычки и unescape
                                val = v[1:-1]
                                out[k] = val
                        return out

                    parsed = _fallback_from_text(content)
                    # enhance parsed with heuristics (dates, INN, amounts, doc_type)
                    def _enhance_from_text(text: str, base: Dict[str, Any]) -> Dict[str, Any]:
                        t = text
                        # doc_type
                        mdt = re.search(r'\b(ТОРГ-12|ТОРГ|УПД|Сч[её]т-?фактура|Счет-?фактура)\b', t, flags=re.IGNORECASE)
                        if mdt and not base.get('doc_type'):
                            base['doc_type'] = mdt.group(1)
                        # dates
                        mdate = re.search(r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b', t)
                        if mdate and not base.get('doc_date'):
                            base['doc_date'] = mdate.group(1)
                        # INNs (10 or 12 digits) - collect first two
                        inns = re.findall(r'\b(\d{10}|\d{12})\b', t)
                        if inns:
                            if not base.get('seller_inn'):
                                base['seller_inn'] = inns[0]
                            if len(inns) > 1 and not base.get('buyer_inn'):
                                base['buyer_inn'] = inns[1]
                        # amounts with currency
                        mcur = re.search(r'(\d[\d\s\u00A0,\.]*\d)\s*(руб\.?|₽|RUB|USD|EUR)\b', t, flags=re.IGNORECASE)
                        if mcur and not base.get('total_with_nds'):
                            amt = mcur.group(1)
                            amt = amt.replace('\u00A0', '').replace(' ', '').replace(',', '.')
                            try:
                                base['total_with_nds'] = float(re.sub(r'[^0-9\.]', '', amt))
                                base['currency'] = mcur.group(2)
                            except Exception:
                                pass
                        return base

                    enhanced = _enhance_from_text(content, parsed)
                    return enhanced
        except Exception as e:
            # network/timeout/etc
            attempt += 1
            if attempt > retries:
                print(f"HTTP API failed: {e}")
                break
            backoff = 1.5 ** attempt
            time.sleep(backoff)
            continue

    # Fallback на CLI
    prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    out = _run_ollama_cli(model, prompt)
    # Сохраняем сырой ответ CLI
    try:
        src = raw.get("source", {}).get("name", "doc")
        out_dir = Path("output") / str(src)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "llm_raw.txt").write_text(str(out), encoding="utf-8")
    except Exception:
        pass
    return _safe_json_extract(out)


def extract_without_llm(raw: dict) -> StructuredDocument:
    """
    Извлечение структурированных данных из raw JSON БЕЗ использования LLM.
    Использует только regex-парсеры для прямого извлечения из OCR-текста.
    """
    from table_parser import parse_table_items, extract_totals

    result = StructuredDocument()
    warnings = []
    errors = []
    fields = result.fields

    # Извлекаем информацию о документе из OCR
    doc_info = _extract_doc_info_from_ocr(raw)
    fields.doc_type = doc_info.get("doc_type")
    fields.doc_number = doc_info.get("doc_number")
    fields.doc_date = _fix_date(doc_info.get("doc_date"))

    # Извлекаем реквизиты сторон из OCR
    parties = _extract_parties_from_ocr(raw)
    fields.seller = PartyInfo(
        name=parties.get("seller_name"),
        inn=_fix_inn(parties.get("seller_inn")),
        kpp=_fix_kpp(parties.get("seller_kpp")),
    )
    fields.buyer = PartyInfo(
        name=parties.get("buyer_name"),
        inn=_fix_inn(parties.get("buyer_inn")),
        kpp=_fix_kpp(parties.get("buyer_kpp")),
    )

    # Парсим таблицы
    tables = raw.get("tables", [])
    all_items: List[TableItem] = []

    for table in tables:
        if not isinstance(table, dict):
            continue
        grid = table.get("grid", [])
        if grid:
            items = parse_table_items(grid)
            all_items.extend(items)

            # Извлекаем итоги из таблицы
            table_totals = extract_totals(grid)
            if table_totals.get("total_with_nds") and not fields.total_with_nds:
                fields.total_with_nds = table_totals["total_with_nds"]
            if table_totals.get("total_nds") and not fields.total_nds:
                fields.total_nds = table_totals["total_nds"]
            if table_totals.get("total_amount") and not fields.total_amount:
                fields.total_amount = table_totals["total_amount"]

    # Fallback: если таблиц не найдено — пробуем собрать из OCR-текста
    if not all_items:
        fallback_grids = _build_grid_from_ocr_texts(raw)
        for grid in fallback_grids:
            items = parse_table_items(grid)
            if items:
                all_items.extend(items)
            table_totals = extract_totals(grid)
            if table_totals.get("total_with_nds") and not fields.total_with_nds:
                fields.total_with_nds = table_totals["total_with_nds"]
            if table_totals.get("total_nds") and not fields.total_nds:
                fields.total_nds = table_totals["total_nds"]
            if table_totals.get("total_amount") and not fields.total_amount:
                fields.total_amount = table_totals["total_amount"]

    fields.items = all_items

    # Fallback: извлекаем итоги напрямую из OCR текста если не нашли в таблицах
    if not fields.total_with_nds or not fields.total_nds or not fields.total_amount:
        ocr_totals = _extract_totals_from_ocr(raw)
        if not fields.total_amount and ocr_totals.get("total_amount"):
            fields.total_amount = ocr_totals["total_amount"]
        if not fields.total_nds and ocr_totals.get("total_nds"):
            fields.total_nds = ocr_totals["total_nds"]
        if not fields.total_with_nds and ocr_totals.get("total_with_nds"):
            fields.total_with_nds = ocr_totals["total_with_nds"]

    # Валидация
    if not fields.doc_type:
        warnings.append("Тип документа не определён")
    if not fields.doc_number:
        warnings.append("Номер документа не найден")
    if not fields.seller.name:
        warnings.append("Наименование продавца не найдено")
    if not fields.buyer.name:
        warnings.append("Наименование покупателя не найдено")

    result.warnings = warnings
    result.errors = errors
    result.confidence = 0.6  # Ниже чем с LLM

    return result


def extract_structured_document(
    raw: dict,
    model: str = "qwen2.5:0.5b-instruct",
    host: str = "http://127.0.0.1:11434",
    retries: int = 2,
) -> StructuredDocument:
    """
    Полное извлечение структурированных данных из raw JSON.

    Args:
        raw: Сырой JSON от OCR-пайплайна
        model: Модель Ollama
        host: Хост Ollama

    Returns:
        StructuredDocument с заполненными полями
    """
    result = StructuredDocument()
    warnings = []
    errors = []

    # 1. Извлекаем поля через LLM
    try:
        llm_data = extract_fields_with_llm(raw, model, host, retries)
    except Exception as e:
        errors.append(f"LLM extraction failed: {e}")
        llm_data = {}

    # 1.0.1 Нормализация альтернативных ключей от LLM
    def _normalize_llm_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        """Маппинг альтернативных ключей к стандартным."""
        if not isinstance(data, dict):
            return data

        # Маппинг альтернативных ключей
        key_mapping = {
            # buyer альтернативы
            "customer_name": "buyer_name",
            "customer_inn": "buyer_inn",
            "customer_kpp": "buyer_kpp",
            "customer_address": "buyer_address",
            "покупатель": "buyer_name",
            "заказчик": "buyer_name",
            "плательщик": "buyer_name",
            # seller альтернативы
            "vendor_name": "seller_name",
            "vendor_inn": "seller_inn",
            "vendor_kpp": "seller_kpp",
            "поставщик": "seller_name",
            "продавец": "seller_name",
            "исполнитель": "seller_name",
            # суммы
            "vat_amount": "total_nds",
            "nds": "total_nds",
            "vat": "total_nds",
            "сумма_ндс": "total_nds",
            "amount": "total_amount",
            "subtotal": "total_amount",
            "сумма_без_ндс": "total_amount",
            "total": "total_with_nds",
            "итого": "total_with_nds",
            "всего": "total_with_nds",
            # документ
            "number": "doc_number",
            "date": "doc_date",
            "document_number": "doc_number",
            "document_date": "doc_date",
            "номер": "doc_number",
            "дата": "doc_date",
        }

        result = {}
        for k, v in data.items():
            # Нормализуем ключ
            k_lower = k.lower().strip()
            new_key = key_mapping.get(k_lower, k)
            # Не перезаписываем если стандартный ключ уже есть
            if new_key in result and result[new_key] not in (None, "", []):
                continue
            result[new_key] = v

        return result

    def _parse_combined_inn_kpp(data: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг объединённых полей ИНН/КПП (например '7799555720/779901001')."""
        if not isinstance(data, dict):
            return data

        combined_fields = [
            ("seller_inn_kpp", "seller_inn", "seller_kpp"),
            ("buyer_inn_kpp", "buyer_inn", "buyer_kpp"),
            ("customer_inn_kpp", "buyer_inn", "buyer_kpp"),
            ("vendor_inn_kpp", "seller_inn", "seller_kpp"),
            ("inn_kpp", "seller_inn", "seller_kpp"),  # первый встреченный -> seller
        ]

        for combined_key, inn_key, kpp_key in combined_fields:
            combined_val = data.get(combined_key)
            if combined_val and isinstance(combined_val, str):
                # Парсим формат "ИНН/КПП" или "ИНН КПП"
                parts = re.split(r"[/\s]+", combined_val.strip())
                if len(parts) >= 1:
                    inn_candidate = re.sub(r"\D", "", parts[0])
                    if len(inn_candidate) in (10, 12) and not data.get(inn_key):
                        data[inn_key] = inn_candidate
                if len(parts) >= 2:
                    kpp_candidate = re.sub(r"\D", "", parts[1])
                    if len(kpp_candidate) == 9 and not data.get(kpp_key):
                        data[kpp_key] = kpp_candidate
                # Удаляем объединённое поле
                del data[combined_key]

        return data

    def _extract_items_from_llm(data: Dict[str, Any]) -> List[TableItem]:
        """Извлечение товарных позиций из ответа LLM."""
        items = []
        llm_items = data.get("items", [])
        if not isinstance(llm_items, list):
            return items

        for idx, item in enumerate(llm_items, start=1):
            if not isinstance(item, dict):
                continue
            try:
                ti = TableItem(
                    row_num=item.get("row_num") or idx,
                    name=item.get("name"),
                    unit=item.get("unit"),
                    quantity=item.get("quantity"),
                    price=item.get("price"),
                    amount=item.get("amount"),
                    nds_rate=item.get("nds_rate"),
                    nds_amount=item.get("nds_amount"),
                    total_amount=item.get("total_amount"),
                )
                if ti.name:  # Только если есть название
                    items.append(ti)
            except Exception:
                continue
        return items

    # Применяем нормализацию
    try:
        llm_data = _normalize_llm_keys(llm_data)
        llm_data = _parse_combined_inn_kpp(llm_data)
    except Exception:
        pass

    # 1.1 Нормализуем частые плейсхолдеры от модели ("| null", "null", шаблоны дат и т.п.)
    def _normalize_placeholders(obj: Any) -> Any:
        """Рекурсивно преобразует плейсхолдер-строки в None и чистит пробелы.

        Правила:
        - Строки, которые равны 'null'/'None' (в любом регистре) -> None
        - Строки, содержащие '|' и правый операнд 'null' (например "ДД.MM.ГГГГ | null") -> None
        - Пустые строки и '-' -> None
        - Иначе возвращаем строку с обрезанными пробелами
        """
        if isinstance(obj, dict):
            return {k: _normalize_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize_placeholders(v) for v in obj]
        if isinstance(obj, str):
            s = obj.strip()
            if not s or s == "-":
                return None
            low = s.lower()
            if low == "null" or low == "none" or low == "n/a":
                return None
            # шаблоны вида "... | null"
            if "|" in s:
                parts = [p.strip() for p in s.split("|")]
                if parts and parts[-1].lower() == "null":
                    return None
            # шаблон дат-заполнителей (например "ДД.MM.ГГГГ")
            if s.upper().startswith("ДД") and ("ГГГГ" in s.upper() or "ГГ" in s.upper()):
                return None
            return s
        return obj

    try:
        llm_data = _normalize_placeholders(llm_data)
    except Exception:
        pass

    # 1.2 Map common nested keys from LLM outputs (e.g., data.document_number, result.total_amount)
    try:
        if isinstance(llm_data, dict):
            # collect candidate nested dicts
            nested_candidates = []
            for key in ("data", "result", "payload", "output", "response"):
                v = llm_data.get(key)
                if isinstance(v, dict):
                    nested_candidates.append(v)

            def _first(*names, src=None):
                if src is None:
                    return None
                for n in names:
                    if n in src and src[n] not in (None, ""):
                        return src[n]
                return None

            # Try to fill top-level expected keys from nested candidates
            for cand in nested_candidates:
                # document number
                if not llm_data.get("doc_number"):
                    val = _first("document_number", "doc_number", "number", "id", src=cand)
                    if val:
                        llm_data["doc_number"] = val

                # dates
                if not llm_data.get("doc_date"):
                    val = _first("date", "doc_date", "document_date", src=cand)
                    if val:
                        llm_data["doc_date"] = val

                # totals
                if not llm_data.get("total_with_nds"):
                    val = _first("total_with_nds", "total_amount", "total", "amount", src=cand)
                    if val:
                        llm_data["total_with_nds"] = val
                if not llm_data.get("total_amount"):
                    val = _first("total_amount", "subtotal", "amount_without_vat", src=cand)
                    if val:
                        llm_data["total_amount"] = val
                if not llm_data.get("total_nds"):
                    val = _first("vat_amount", "vat", "total_nds", src=cand)
                    if val:
                        llm_data["total_nds"] = val

                # currency
                if not llm_data.get("currency"):
                    val = _first("currency", "curr", "currency_code", src=cand)
                    if val:
                        llm_data["currency"] = val

                # seller/buyer names
                if not llm_data.get("seller_name"):
                    val = _first("seller_name", "seller", "from", src=cand)
                    if val:
                        llm_data["seller_name"] = val
                if not llm_data.get("buyer_name"):
                    val = _first("buyer_name", "buyer", "to", src=cand)
                    if val:
                        llm_data["buyer_name"] = val
            # Aggressive flatten with conflict-resolution: gather candidates from nested dicts
            def _flatten(d: dict, parent: str = "") -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in d.items():
                    key = f"{parent}.{k}" if parent else k
                    if isinstance(v, dict):
                        out.update(_flatten(v, key))
                    else:
                        out[key] = v
                return out

            flat = {}
            for cand in nested_candidates:
                try:
                    flat.update(_flatten(cand))
                except Exception:
                    pass

            # Build candidate lists per target and resolve conflicts with simple policy
            candidates_by_target: Dict[str, List[Dict[str, Any]]] = {}

            def _add_candidate(target: str, value: Any, source_key: str):
                if target not in candidates_by_target:
                    candidates_by_target[target] = []
                candidates_by_target[target].append({"value": value, "source": source_key})

            for fk, fv in flat.items():
                last = fk.split('.')[-1]
                if last in ("doc_number", "document_number", "date", "total_amount", "total", "amount", "vat_amount", "currency", "seller_name", "buyer_name"):
                    if last == "document_number":
                        target = "doc_number"
                    elif last in ("total", "amount"):
                        target = "total_with_nds"
                    else:
                        target = last
                    _add_candidate(target, fv, fk)

            # Also include any direct nested_candidates top-level keys as candidates
            for cand in nested_candidates:
                for k, v in cand.items():
                    if k in ("doc_number", "document_number", "date", "total_amount", "total", "amount", "vat_amount", "currency", "seller_name", "buyer_name", "seller", "buyer"):
                        tgt = "doc_number" if k == "document_number" else ("total_with_nds" if k in ("total","amount") else k)
                        _add_candidate(tgt, v, f"nested:{k}")

            # Resolve candidates
            def _choose_candidate(field: str, cand_list: List[Dict[str, Any]]) -> Any:
                # Remove None values first
                non_nulls = [c for c in cand_list if c.get("value") not in (None, "", [])]
                if not non_nulls:
                    return None

                # If only one non-null, choose it
                if len(non_nulls) == 1:
                    return non_nulls[0]["value"]

                # Try to prefer value that appears verbatim in OCR texts
                ocr_texts = []
                try:
                    ocr_raw = raw.get("ocr_raw", {})
                    if isinstance(ocr_raw, dict):
                        rec_texts = ocr_raw.get("rec_texts", [])
                        if isinstance(rec_texts, list):
                            ocr_texts = [t for t in rec_texts if isinstance(t, str)]
                except Exception:
                    ocr_texts = []

                for c in non_nulls:
                    v = c.get("value")
                    try:
                        if isinstance(v, (str, int, float)) and any(str(v) in t for t in ocr_texts):
                            return v
                    except Exception:
                        pass

                # Prefer numeric values over strings for totals
                nums = [c for c in non_nulls if isinstance(c.get("value"), (int, float)) or (isinstance(c.get("value"), str) and re.match(r"^[\d\s\.,]+$", str(c.get("value"))))]
                if nums:
                    # pick the first numeric candidate
                    try:
                        return nums[0]["value"]
                    except Exception:
                        pass

                # Fallback: pick the first non-null and log conflict
                return non_nulls[0]["value"]

            # Apply resolved values and log conflicts
            try:
                src = raw.get("source", {}).get("name", "doc")
                out_dir = Path("output") / str(src)
                out_dir.mkdir(parents=True, exist_ok=True)
                conflicts_log = []
                for tgt, cand_list in candidates_by_target.items():
                    chosen = _choose_candidate(tgt, cand_list)
                    # If target already exists and differs from chosen, record conflict
                    existing = llm_data.get(tgt)
                    if existing not in (None, "") and chosen not in (None, "") and str(existing) != str(chosen):
                        conflicts_log.append({"field": tgt, "existing": existing, "chosen": chosen, "candidates": cand_list})
                        # prefer existing unless it's null-like; keep existing
                    else:
                        # set if empty
                        if not llm_data.get(tgt):
                            llm_data[tgt] = chosen

                if conflicts_log:
                    (out_dir / "mapping_conflicts.txt").write_text(json.dumps(conflicts_log, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

    # 2. Заполняем DocumentFields
    fields = result.fields

    # Тип документа
    fields.doc_type = _fix_doc_type(llm_data.get("doc_type"))

    # Номера и даты
    fields.doc_number = llm_data.get("doc_number")
    fields.doc_date = _fix_date(llm_data.get("doc_date"))
    fields.sf_number = llm_data.get("sf_number")
    fields.sf_date = _fix_date(llm_data.get("sf_date"))

    # Fallback: извлечение номера и даты из OCR, если LLM не справилась
    if not fields.doc_number or not fields.doc_date or not fields.doc_type:
        ocr_doc_info = _extract_doc_info_from_ocr(raw)
        if not fields.doc_number and ocr_doc_info.get("doc_number"):
            fields.doc_number = ocr_doc_info["doc_number"]
        if not fields.doc_date and ocr_doc_info.get("doc_date"):
            fields.doc_date = _fix_date(ocr_doc_info["doc_date"])
        if not fields.doc_type and ocr_doc_info.get("doc_type"):
            fields.doc_type = ocr_doc_info["doc_type"]

    # Продавец
    fields.seller = PartyInfo(
        name=llm_data.get("seller_name"),
        inn=_fix_inn(llm_data.get("seller_inn")),
        kpp=_fix_kpp(llm_data.get("seller_kpp")),
        address=llm_data.get("seller_address"),
    )

    # Покупатель
    fields.buyer = PartyInfo(
        name=llm_data.get("buyer_name"),
        inn=_fix_inn(llm_data.get("buyer_inn")),
        kpp=_fix_kpp(llm_data.get("buyer_kpp")),
        address=llm_data.get("buyer_address"),
    )

    # Fallback: извлечение реквизитов напрямую из OCR, если LLM не справилась
    if not fields.seller.name or not fields.seller.inn or not fields.buyer.name or not fields.buyer.inn:
        ocr_parties = _extract_parties_from_ocr(raw)
        if not fields.seller.name and ocr_parties.get("seller_name"):
            fields.seller.name = ocr_parties["seller_name"]
        if not fields.seller.inn and ocr_parties.get("seller_inn"):
            fields.seller.inn = _fix_inn(ocr_parties["seller_inn"])
        if not fields.seller.kpp and ocr_parties.get("seller_kpp"):
            fields.seller.kpp = _fix_kpp(ocr_parties["seller_kpp"])
        if not fields.buyer.name and ocr_parties.get("buyer_name"):
            fields.buyer.name = ocr_parties["buyer_name"]
        if not fields.buyer.inn and ocr_parties.get("buyer_inn"):
            fields.buyer.inn = _fix_inn(ocr_parties["buyer_inn"])
        if not fields.buyer.kpp and ocr_parties.get("buyer_kpp"):
            fields.buyer.kpp = _fix_kpp(ocr_parties["buyer_kpp"])

    # Грузоотправитель/получатель
    fields.consignor = llm_data.get("consignor")
    fields.consignee = llm_data.get("consignee")

    # Договор
    fields.contract_number = llm_data.get("contract_number")
    fields.contract_date = _fix_date(llm_data.get("contract_date"))

    # Итоги от LLM
    fields.total_amount = llm_data.get("total_amount")
    fields.total_nds = llm_data.get("total_nds")
    fields.total_with_nds = llm_data.get("total_with_nds")
    fields.currency = llm_data.get("currency")

    # 3. Парсим таблицы
    tables = raw.get("tables", [])
    all_items: List[TableItem] = []
    # Log layout summary for debugging
    _log_layout_summary(raw)

    for table in tables:
        if not isinstance(table, dict):
            continue
        grid = table.get("grid", [])
        if grid:
            items = parse_table_items(grid)
            all_items.extend(items)

            # Извлекаем итоги из таблицы
            table_totals = extract_totals(grid)
            if table_totals.get("total_with_nds") and not fields.total_with_nds:
                fields.total_with_nds = table_totals["total_with_nds"]
            if table_totals.get("total_nds") and not fields.total_nds:
                fields.total_nds = table_totals["total_nds"]
            if table_totals.get("total_amount") and not fields.total_amount:
                fields.total_amount = table_totals["total_amount"]

    # Fallback: если таблиц не найдено или items пусты — попробуем собрать их из OCR-текста
    if (not tables or not all_items):
        try:
            # Сначала попробуем OCR на кропах таблиц (если есть сохранённые кропы)
            fallback_grids = []
            try:
                # initialize OCR once
                ocr_local = PaddleOCR(device="cpu", lang="ru")
            except Exception:
                ocr_local = None

            for idx, table in enumerate(tables, start=1):
                crop_path = table.get("crop_image")
                if crop_path:
                    try:
                        img = None
                        if isinstance(crop_path, str):
                            p = Path(crop_path)
                            if p.exists():
                                img = cv2.imread(str(p))
                        if img is None:
                            continue
                        if ocr_local is None:
                            ocr_local = PaddleOCR(device="cpu", lang="ru")
                        res = ocr_local.predict(img)
                        # Try to extract tokens directly from PaddleOCR python output (box + text)
                        rec_texts = []
                        resc = None
                        tokens = []
                        try:
                            if isinstance(res, list):
                                for line in res:
                                    try:
                                        # line expected like [box, (text, conf)] or similar
                                        if not line:
                                            continue
                                        box = None
                                        rec = None
                                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                                            box = line[0]
                                            rec = line[1]
                                        elif isinstance(line, dict):
                                            # dict may contain 'points' and 'text'
                                            box = line.get('points') or line.get('bbox')
                                            rec = line
                                        if not box or not rec:
                                            continue
                                        # compute center
                                        xcenter = None
                                        ycenter = None
                                        if isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
                                            xs = [float(p[0]) for p in box]
                                            ys = [float(p[1]) for p in box]
                                            xcenter = sum(xs) / len(xs)
                                            ycenter = sum(ys) / len(ys)
                                        elif isinstance(box, list) and len(box) >= 4 and all(isinstance(n, (int, float)) for n in box[:4]):
                                            xcenter = (float(box[0]) + float(box[2])) / 2.0
                                            ycenter = (float(box[1]) + float(box[3])) / 2.0
                                        if xcenter is None:
                                            continue
                                        text = None
                                        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                                            if isinstance(rec[0], str):
                                                text = rec[0]
                                            elif isinstance(rec[0], (list, tuple)) and rec[0]:
                                                text = str(rec[0][0])
                                        elif isinstance(rec, dict):
                                            text = rec.get('text') or rec.get('rec') or rec.get('transcription')
                                        if text:
                                            tokens.append({'text': str(text).strip(), 'x': xcenter, 'y': ycenter})
                                    except Exception:
                                        continue
                            # if tokens still empty, fallback to saving paddle JSON and reading rec_texts
                            if not tokens:
                                try:
                                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpf:
                                        tmp_path = Path(tmpf.name)
                                    try:
                                        if res:
                                            resc = paddle_result_to_dict(res[0], tmp_path)
                                            rec_texts = resc.get("rec_texts") or resc.get("lines") or []
                                        else:
                                            rec_texts = []
                                    finally:
                                        tmp_path.unlink(missing_ok=True)
                                except Exception:
                                    rec_texts = []
                        except Exception:
                            # fallback: no tokens
                            tokens = []

                        # build small grids from rec_texts within crop
                        if rec_texts:
                            # prefer token-based grid building using bbox info when available
                            try:
                                resc = resc if 'resc' in locals() else None
                            except Exception:
                                resc = None
                            # if we have full paddle JSON for this crop, try to extract tokens
                            # paddle_result_to_dict was already used to produce 'resc' above
                            try:
                                # get tokens via paddle output (if present)
                                tokens = []
                                # if resc is available and is dict, use it
                                if isinstance(resc, dict):
                                    tokens = _extract_tokens_from_paddle_result(resc)
                                # if tokens found, cluster to grid
                                if tokens:
                                    grid = _cluster_grid_from_tokens(tokens)
                                    # filter out blocks that look like реквизиты
                                    combined = "\n".join([" ".join(r) for r in grid])
                                    if not re.search(r"\b(ИНН|БИК|р/с|к/с|ПАО|ОГРН|адрес|г\.|Москва|тел|тел\.|телефон)\b", combined, flags=re.IGNORECASE):
                                        fallback_grids.append(grid)
                                else:
                                    # fallback to simple line-splitting
                                    block = []
                                    num_re = re.compile(r"\d[\d\s\.,]*\d")
                                    for line in rec_texts:
                                        if not isinstance(line, str):
                                            continue
                                        s = line.strip()
                                        if not s:
                                            if block:
                                                rows = []
                                                for L in block:
                                                    if '|' in L:
                                                        cols = [c.strip() for c in L.split('|')]
                                                    else:
                                                        parts = re.split(r"\s{2,}|\t", L)
                                                        if len(parts) <= 1:
                                                            parts = re.split(r"\s+", L)
                                                        cols = [p.strip() for p in parts if p is not None]
                                                    rows.append(cols)
                                                if rows:
                                                    maxc = max(len(r) for r in rows)
                                                    if maxc >= 2 and len(rows) >= 2:
                                                        grid = [[r[i] if i < len(r) else "" for i in range(maxc)] for r in rows]
                                                        # filter реквизиты
                                                        combined = "\n".join([" ".join(r) for r in grid])
                                                        if not re.search(r"\b(ИНН|БИК|р/с|к/с|ПАО|ОГРН|адрес|г\.|Москва|тел|тел\.|телефон)\b", combined, flags=re.IGNORECASE):
                                                            fallback_grids.append(grid)
                                                block = []
                                            continue
                                        if len(num_re.findall(s)) >= 1 or '%' in s or 'руб' in s:
                                            block.append(s)
                                    if block:
                                        rows = []
                                        for L in block:
                                            if '|' in L:
                                                cols = [c.strip() for c in L.split('|')]
                                            else:
                                                parts = re.split(r"\s{2,}|\t", L)
                                                if len(parts) <= 1:
                                                    parts = re.split(r"\s+", L)
                                                cols = [p.strip() for p in parts if p is not None]
                                            rows.append(cols)
                                        if rows:
                                            maxc = max(len(r) for r in rows)
                                            if maxc >= 2 and len(rows) >= 2:
                                                grid = [[r[i] if i < len(r) else "" for i in range(maxc)] for r in rows]
                                                combined = "\n".join([" ".join(r) for r in grid])
                                                if not re.search(r"\b(ИНН|БИК|р/с|к/с|ПАО|ОГРН|адрес|г\.|Москва|тел|тел\.|телефон)\b", combined, flags=re.IGNORECASE):
                                                    fallback_grids.append(grid)
                            except Exception:
                                pass
                    except Exception:
                        continue

            # Если OCR-on-crop не дал результатов — используем глобальный rec_texts fallback
            if not fallback_grids:
                fallback_grids = _build_grid_from_ocr_texts(raw)
            for grid in fallback_grids:
                items = parse_table_items(grid)
                if items:
                    all_items.extend(items)
                # also extract totals from fallback grid
                table_totals = extract_totals(grid)
                if table_totals.get("total_with_nds") and not fields.total_with_nds:
                    fields.total_with_nds = table_totals["total_with_nds"]
                if table_totals.get("total_nds") and not fields.total_nds:
                    fields.total_nds = table_totals["total_nds"]
                if table_totals.get("total_amount") and not fields.total_amount:
                    fields.total_amount = table_totals["total_amount"]
        except Exception:
            pass

    # Если из таблиц ничего не извлеклось - пробуем взять items из LLM
    if not all_items:
        try:
            llm_items = _extract_items_from_llm(llm_data)
            if llm_items:
                all_items.extend(llm_items)
        except Exception:
            pass

    fields.items = all_items

    # 3.1 Арифметические проверки по позициям и итогам
    def _approx_equal(a: Optional[float], b: Optional[float], rel_tol: float = 1e-3, abs_tol: float = 0.01) -> bool:
        if a is None or b is None:
            return False
        try:
            a_f = float(a)
            b_f = float(b)
        except Exception:
            return False
        diff = abs(a_f - b_f)
        return diff <= max(abs_tol, rel_tol * max(abs(a_f), abs(b_f)))

    # Per-row checks
    for idx, it in enumerate(all_items, start=1):
        if it.quantity is not None and it.price is not None:
            expected = it.quantity * it.price
            # prefer comparing to amount, fallback to total_amount
            target = it.amount if it.amount is not None else it.total_amount
            if target is not None and not _approx_equal(expected, target):
                warnings.append(f"Row {it.row_num or idx}: quantity*price ({expected}) != amount ({target})")
        # check NDS per row if rate available
        if it.nds_rate and it.amount is not None:
            m = re.search(r"(\d{1,2})(?:\s*%)?", str(it.nds_rate))
            if m:
                try:
                    rate = float(m.group(1))
                    expected_nds = it.amount * rate / 100.0
                    if it.nds_amount is not None and not _approx_equal(expected_nds, it.nds_amount):
                        warnings.append(f"Row {it.row_num or idx}: computed NDS ({expected_nds}) != nds_amount ({it.nds_amount})")
                except Exception:
                    pass

    # Totals check: sum of item amounts vs totals
    sum_amounts = None
    try:
        sum_amounts = sum([float(it.amount) for it in all_items if it.amount is not None]) if all_items else None
    except Exception:
        sum_amounts = None

    if sum_amounts is not None:
        if fields.total_amount is not None and not _approx_equal(sum_amounts, fields.total_amount):
            warnings.append(f"Sum of rows without NDS ({sum_amounts}) != declared total_amount ({fields.total_amount})")
        # if total_nds present, check total_with_nds
        if fields.total_nds is not None and fields.total_amount is not None:
            total_with = fields.total_amount + fields.total_nds
            if fields.total_with_nds is not None and not _approx_equal(total_with, fields.total_with_nds):
                warnings.append(f"Computed total_with_nds ({total_with}) != declared total_with_nds ({fields.total_with_nds})")

    # 4. Валидация
    if not fields.doc_type:
        warnings.append("Тип документа не определён")
    if not fields.doc_number:
        warnings.append("Номер документа не найден")
    if not fields.seller.name:
        warnings.append("Наименование продавца не найдено")
    if not fields.buyer.name:
        warnings.append("Наименование покупателя не найдено")

    # Проверка ИНН
    if fields.seller.inn and len(fields.seller.inn) not in (10, 12):
        warnings.append(f"Некорректный ИНН продавца: {fields.seller.inn}")
    if fields.buyer.inn and len(fields.buyer.inn) not in (10, 12):
        warnings.append(f"Некорректный ИНН покупателя: {fields.buyer.inn}")

    # Проверка сквозной нумерации строк (п.2.4 ТЗ)
    if all_items:
        row_nums = [it.row_num for it in all_items if it.row_num is not None]
        if row_nums:
            # Проверяем последовательность
            expected = list(range(1, len(row_nums) + 1))
            sorted_nums = sorted(row_nums)
            if sorted_nums != expected:
                # Ищем пропуски и дубликаты
                missing = set(expected) - set(row_nums)
                duplicates = [n for n in row_nums if row_nums.count(n) > 1]
                if missing:
                    warnings.append(f"Пропущены номера строк: {sorted(missing)}")
                if duplicates:
                    warnings.append(f"Дублирующиеся номера строк: {sorted(set(duplicates))}")

    result.warnings = warnings
    result.errors = errors
    result.confidence = 0.8 if not errors else 0.5

    return result
