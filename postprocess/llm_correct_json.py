import argparse
import os
import re
import unicodedata
from gpt4all import GPT4All

# ====== Фильтры для "говно-слов" ======

RUS_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def vowel_ratio(word: str) -> float:
    word = ''.join(ch for ch in word if ch.isalpha())
    if not word:
        return 0.0
    v = sum(1 for c in word if c in RUS_VOWELS)
    return v / len(word)


def is_gibberish(word: str) -> bool:
    w = word.strip()
    if len(w) <= 3:
        return False

    # много согласных подряд
    if re.search(r"[бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ]{5,}", w):
        return True

    # мало гласных
    if vowel_ratio(w) < 0.15:
        return True

    return False


def has_script_mix(word: str) -> bool:
    """Есть ли перемешивание латиницы и кириллицы в слове."""
    scripts = set()
    for c in word:
        name = unicodedata.name(c, "")
        if "CYRILLIC" in name:
            scripts.add("CYR")
        elif "LATIN" in name:
            scripts.add("LAT")
    return len(scripts) >= 2


LATIN_TO_CYR = {
    "A": "А", "a": "а",
    "B": "В", "E": "Е", "e": "е",
    "K": "К", "M": "М",
    "H": "Н", "O": "О", "o": "о",
    "P": "Р", "C": "С", "c": "с",
    "T": "Т", "X": "Х",
    "Y": "У", "y": "у",
    "N": "Н",  # иногда путают
    # и т.д. по желанию
}

def normalize_mixed_scripts(text: str) -> str:
    res = []
    for ch in text:
        if ch in LATIN_TO_CYR:
            res.append(LATIN_TO_CYR[ch])
        else:
            res.append(ch)
    return "".join(res)



def need_llm_fix(text: str) -> bool:
    s = text.strip()
    if not s:
        return False

    # выкинем совсем короткие / слишком длинные
    if len(s) < 5:
        return False
    if len(s) > 200:
        # длинные монструозные строки лучше не мучить
        return False

    tokens = [t for t in s.split() if t]
    long_tokens = [t for t in tokens if len(t) >= 4]
    if not long_tokens:
        return False

    # 1) если есть слово со смешанной кириллицей/латиницей → сразу в ЛЛМ
    if any(has_script_mix(t) for t in long_tokens):
        return True

    # 2) доля "каши"
    gib = sum(1 for t in long_tokens if is_gibberish(t))
    ratio = gib / len(long_tokens)
    if ratio >= 0.20:
        return True

    # 3) средняя доля гласных
    vr_vals = [vowel_ratio(t) for t in long_tokens]
    avg_vr = sum(vr_vals) / len(vr_vals)
    if avg_vr < 0.25:
        return True

    return False


# ====== ЛЛМ ======

MODEL_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"

SYSTEM_PROMPT = (
    "Ты исправляешь строку после OCR в русском бухгалтерском документе.\n"
    "Тебе даётся одна короткая строка с ошибками распознавания.\n"
    "Нужно вернуть ТОЛЬКО исправленную строку, без комментариев, без пояснений,\n"
    "без слов 'Подсказка', 'Пример', 'Ответ', без кавычек.\n"
    "Не сокращай фразу и не добавляй новый смысл.\n"
)


BAD_GENERIC = {"ТОЛЬКО", "Только", "только"}

def clean_llm_output(raw: str, original: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return original

    cand = lines[0]

    # если модель сгенерировала "Подсказка..." и т.п. — сразу в корзину
    bad_prefixes = (
        "Подсказка",
        "Пример",
        "Ответ",
        "Текст с ошибками",
        "Нормальный русский текст",
        "Вы действуете",
        "Ты действуешь",
        "Привет",
    )
    if any(cand.startswith(p) for p in bad_prefixes):
        return original

    # если ответ — одно слово из чёрного списка
    if cand.upper() in BAD_GENERIC:
        return original

    # если ответ подозрительно короткий относительно оригинала
    if len(cand) < max(3, int(0.5 * len(original))):
        return original

    # если в оригинале были цифры, а в ответе их нет — тоже подозрительно
    if any(ch.isdigit() for ch in original) and not any(ch.isdigit() for ch in cand):
        return original

    return cand



def correct_with_llm(model: GPT4All, text: str) -> str:
    prompt = (
        SYSTEM_PROMPT
        + "\nСтрока с ошибками OCR:\n"
        + text.strip()
        + "\n\nИсправленная строка:\n"
    )
    raw = model.generate(prompt, max_tokens=80, temp=0.1, top_p=0.9)
    fixed = clean_llm_output(raw, text)
    return fixed


# ====== Основная логика замены в JSON-подобном файле ======

def process_file(input_path: str, output_path: str, model_name: str):
    if not os.path.exists(input_path):
        print("Не найден входной файл:", input_path)
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем пары вида: "tr...что-то...": "ТЕКСТ"
    # Ключи у тебя типа "trаnsсriрtiоn" с кириллицей, поэтому ловим всё, что начинается на "tr"
    pattern = re.compile(r'("tr[^"]*"\s*:\s*")([^"]*)(")')

    matches = list(pattern.finditer(content))
    print(f"Найдено transcription-подобных полей: {len(matches)}")

    if not matches:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Совпадений нет, просто скопировал файл.")
        return

    print("Загружаю модель:", model_name)
    model = GPT4All(model_name)

    # Собираем новый текст по кускам, чтобы не возиться с индексами вручную
    new_parts = []
    last_idx = 0
    fixed_count = 0
    tried_count = 0

    for m in matches:
        start, end = m.span()
        prefix, old_text, suffix = m.group(1), m.group(2), m.group(3)

        # добавляем кусок до этого match
        new_parts.append(content[last_idx:start])

        text = normalize_mixed_scripts(text.strip())
        if need_llm_fix(text):
            tried_count += 1
            try:
                new_text = correct_with_llm(model, text)
                if new_text != text:
                    fixed_count += 1
                    print(f"[FIX] '{text}'  ->  '{new_text}'")
                else:
                    print(f"[SKIP_SAME] '{text}'")
            except Exception as e:
                print(f"[ERROR_LLM] '{text}' -> {e}")
                new_text = text
        else:
            new_text = text

        # сохраняем с теми же кавычками и ключом
        new_parts.append(prefix + new_text + suffix)
        last_idx = end

    # добавляем хвост файла
    new_parts.append(content[last_idx:])
    new_content = "".join(new_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Готово. Всего полей: {len(matches)}, отправили в ЛЛМ: {tried_count}, реально изменили: {fixed_count}")
    print("Результат сохранён в:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="исходный rec_results_post.txt (JSON-подобный)")
    parser.add_argument("--output", required=True, help="куда сохранить файл с исправленными transcription")
    parser.add_argument("--model", default=MODEL_NAME, help="имя/файл модели GPT4All (.gguf)")
    args = parser.parse_args()

    process_file(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
