# postprocess/llm_structurer.py
from __future__ import annotations

import json
import re
import subprocess
from typing import Optional, List, Any

import requests
from pydantic import BaseModel, Field


# ----------------- SCHEMA -----------------

class DocFields(BaseModel):
    doc_type: Optional[str] = None
    doc_number: Optional[str] = None
    doc_date: Optional[str] = None
    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None
    total_sum: Optional[float] = None
    currency: Optional[str] = None


class StructuredResult(BaseModel):
    fields: DocFields
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    notes: List[str] = []


# ----------------- UTILS -----------------

def _safe_json_extract(text: str) -> Any:
    """
    Вытаскиваем JSON даже если модель добавила мусор вокруг.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in model output. First 200 chars: {text[:200]!r}")
    return json.loads(m.group(0))

def _normalize_to_structured(data: Any) -> dict:
    """
    Приводим ответ модели к формату StructuredResult:
    - либо уже {"fields": {...}, "confidence":..., "notes":[...]}
    - либо плоский {"doc_type":..., ...} -> оборачиваем в {"fields": ...}
    + notes всегда приводим к List[str]
    """
    def normalize_notes(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            out = []
            for item in x:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    # самый частый кейс: {"text": "...", "code": "..."}
                    if "text" in item and isinstance(item["text"], str):
                        out.append(item["text"])
                    else:
                        out.append(json.dumps(item, ensure_ascii=False))
                else:
                    out.append(str(item))
            return out
        # если notes пришёл строкой/объектом
        if isinstance(x, dict):
            if "text" in x and isinstance(x["text"], str):
                return [x["text"]]
            return [json.dumps(x, ensure_ascii=False)]
        return [str(x)]

    if not isinstance(data, dict):
        return {"fields": {}, "confidence": 0.3, "notes": ["Model output was not a dict"]}

    # Уже правильный формат
    if "fields" in data and isinstance(data["fields"], dict):
        data["notes"] = normalize_notes(data.get("notes"))
        data.setdefault("confidence", 0.5)
        return data

    # Плоский формат -> оборачиваем
    known_keys = {"doc_type", "doc_number", "doc_date", "seller_name", "buyer_name", "total_sum", "currency"}
    flat_fields = {k: data.get(k, None) for k in known_keys}

    return {
        "fields": flat_fields,
        "confidence": float(data.get("confidence", 0.5)) if str(data.get("confidence", "")).strip() else 0.5,
        "notes": normalize_notes(data.get("notes")),
    }

def _fix_torg_typo(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    # Частая опечатка латиницей: TORG -> ТОРГ
    s2 = s.replace("TORG", "ТОРГ").replace("TОRG", "ТОРГ").replace("ТОRG", "ТОРГ")
    # Иногда "ТОРG" (латинская G)
    s2 = s2.replace("ТОРG", "ТОРГ")
    return s2


def _postprocess_result(res: StructuredResult) -> StructuredResult:
    """
    3 мини-правила:
    1) doc_type: если где-то фигурирует ТОРГ-12 -> ставим doc_type="ТОРГ-12"
    2) doc_number: чинит латинские буквы в "ТОРG-12" -> "ТОРГ-12"
    3) doc_date: если модель взяла "25.12.98" из постановления -> зануляем и оставляем null
       (чтобы не портить данные), потом извлечём корректно, когда начнём брать строки вокруг "Накладная ... от ..."
    """

    f = res.fields

    # 1) doc_type: нормализуем под ТОРГ-12
    if isinstance(f.doc_type, str) and "ТОРГ" in f.doc_type:
        f.doc_type = "ТОРГ-12"

    # Иногда модель пишет "Унифицированная форма" — но в doc_number уже ТОРГ-12.
    if (not f.doc_type or (isinstance(f.doc_type, str) and "унифицирован" in f.doc_type.lower())) and isinstance(f.doc_number, str):
        dn = _fix_torg_typo(f.doc_number)
        if "ТОРГ-12" in dn or "ТОРГ 12" in dn:
            f.doc_type = "ТОРГ-12"

    # 2) doc_number: исправляем TORG/ТОРG
    f.doc_number = _fix_torg_typo(f.doc_number)

    # 3) doc_date: отсекаем "постановление 25.12.98"
    # Если дата выглядит как **.**.98 или **.**.1998 — скорее всего это не дата накладной.
    if isinstance(f.doc_date, str):
        d = f.doc_date.strip()
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.(98|1998)", d):
            f.doc_date = None

    res.fields = f
    return res


def compact_from_raw(
    raw: dict,
    head_lines: int = 140,
    tail_lines: int = 90,
    max_chars: int = 9000,
) -> str:
    """
    Компактный вход для LLM:
    - берём OCR строки (rec_texts)
    - отдаём HEAD (шапка) + TAIL (итоги/подписи)
    - таблицы grid (если есть)
    """

    parts: List[str] = []

    # ---------- OCR TEXT (head + tail) ----------
    ocr_raw = raw.get("ocr_raw")
    ocr_lines: List[str] = []
    if isinstance(ocr_raw, dict):
        rec_texts = ocr_raw.get("rec_texts") or []
        if isinstance(rec_texts, list):
            for s in rec_texts:
                if isinstance(s, str):
                    s = s.strip()
                    if s:
                        ocr_lines.append(s)

    if ocr_lines:
        head = ocr_lines[:head_lines]
        tail = ocr_lines[-tail_lines:] if len(ocr_lines) > head_lines else []
        text_block = "\n".join(head)
        if tail:
            text_block += "\n...\n" + "\n".join(tail)
        parts.append("OCR TEXT (HEAD+TAIL):\n" + text_block)

    # ---------- TABLES GRID (если не пусто) ----------
    tables = raw.get("tables") or []
    table_blocks: List[str] = []
    if isinstance(tables, list):
        for t in tables:
            if not isinstance(t, dict):
                continue
            grid = t.get("grid") or []
            if isinstance(grid, list) and grid:
                rows_out: List[str] = []
                for row in grid:
                    if isinstance(row, list):
                        row_txt = " | ".join(str(cell).strip() for cell in row)
                        row_txt = row_txt.strip(" |")
                        if row_txt:
                            rows_out.append(row_txt)
                if rows_out:
                    idx = t.get("table_index")
                    table_blocks.append(f"TABLE {idx}:\n" + "\n".join(rows_out))

    if table_blocks:
        parts.append("\n\n".join(table_blocks))

    text = "\n\n".join(parts)
    return text[:max_chars]


def _run_ollama_cli(model: str, prompt: str, timeout_sec: int = 180) -> str:
    """
    Fallback: вызываем ollama CLI, раз `ollama run ...` у тебя работает.
    Это обходит возможные проблемы HTTP-раннера.
    """
    # Важно: prompt передаём как один аргумент (может быть длинным, но у нас он короткий)
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_sec,
    )
    # stderr может содержать полезное
    if proc.returncode != 0:
        raise RuntimeError(f"ollama CLI failed rc={proc.returncode} stderr={proc.stderr[:800]}")
    return proc.stdout.strip()


# ----------------- MAIN -----------------

def structure_with_ollama(
    raw: dict,
    model: str = "qwen2.5:0.5b-instruct",
    host: str = "http://127.0.0.1:11434",
) -> StructuredResult:
    print("USING llm_structurer.py version 2025-12-15")

    system = (
        "You extract fields from OCR text.\n"
        "Return ONLY a valid JSON object (no markdown, no comments).\n"
        "If a field is missing, set it to null.\n"
        "Do not invent values.\n"
        "JSON must match this structure exactly:\n"
        "{"
        "\"fields\":{"
        "\"doc_type\":string|null,"
        "\"doc_number\":string|null,"
        "\"doc_date\":string|null,"
        "\"seller_name\":string|null,"
        "\"buyer_name\":string|null,"
        "\"total_sum\":number|null,"
        "\"currency\":string|null"
        "},"
        "\"confidence\":number,"
        "\"notes\":array"
        "}"
    )

    compact = compact_from_raw(raw)  # намеренно супер-короткий
    user = (
        "Extract structured fields from the document data below.\n"
        "Document data:\n\n"
        + compact
    )

    # ---- Diagnostics: размеры ----
    print("LLM compact length:", len(compact))
    print("LLM compact preview:", compact[:200].replace("\n", "\\n"))

    payload_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 256,
        },
    }
    payload_str = json.dumps(payload_chat, ensure_ascii=False)
    print("Payload bytes (chat):", len(payload_str.encode("utf-8")))

    def finalize(model_text: str) -> StructuredResult:
        data = _safe_json_extract(model_text)
        data = _normalize_to_structured(data)
        res = StructuredResult.model_validate(data)
        return _postprocess_result(res)

    # 1) /api/chat (без format)
    try:
        r = requests.post(f"{host}/api/chat", json=payload_chat, timeout=180)
        if r.ok:
            content = r.json()["message"]["content"]
            return finalize(content)
        else:
            print("HTTP /api/chat failed:", r.status_code, (r.text or "")[:200])
    except Exception as e:
        print("HTTP /api/chat exception:", repr(e))

    # 2) /api/generate fallback
    payload_gen = {
        "model": model,
        "prompt": system + "\n\n" + user,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 256,
        },
    }
    payload_gen_str = json.dumps(payload_gen, ensure_ascii=False)
    print("Payload bytes (generate):", len(payload_gen_str.encode("utf-8")))

    try:
        r2 = requests.post(f"{host}/api/generate", json=payload_gen, timeout=180)
        if r2.ok:
            content = r2.json().get("response", "")
            return finalize(content)
        else:
            print("HTTP /api/generate failed:", r2.status_code, (r2.text or "")[:200])
    except Exception as e:
        print("HTTP /api/generate exception:", repr(e))

    # 3) CLI fallback (должен сработать, раз `ollama run ...` у тебя работает)
    prompt = system + "\n\n" + user
    out_text = _run_ollama_cli(model=model, prompt=prompt, timeout_sec=180)
    return finalize(out_text)