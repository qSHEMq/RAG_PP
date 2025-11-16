import re
import unicodedata
from lexicon import DOMAIN_WORDS  # список доменных слов: "стоимость", "активы", ...

# ===== 1. БАЗОВАЯ НОРМАЛИЗАЦИЯ =====

# визуально похожие латинские -> кириллица
HOMOGRAPH_MAP = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
}

# цифроподобные буквы внутри числовых токенов
DIGIT_LIKE_MAP = {
    "O": "0", "o": "0", "О": "0", "о": "0",
    "З": "3", "з": "3",
    "б": "6", "Б": "6",
    "l": "1", "I": "1",
    "Т": "7", "т": "7",
}

def nkfc(s: str) -> str:
    """Юникод-нормализация."""
    return unicodedata.normalize("NFKC", s)

def map_homographs(s: str) -> str:
    """Заменяем похожие латинские буквы на кириллицу."""
    return "".join(HOMOGRAPH_MAP.get(ch, ch) for ch in s)

# ===== 2. СЛОВАРЬ ТИПОВЫХ ТЕРМИНОВ И ФРАЗ =====

TERMS_CANON = {
    r"\bП[рp]одав[еe]ц\b": "Продавец",
    r"\bП[оo]купател[ьb]\b": "Покупатель",
    r"\bСч[еe][тt]-?ф[аa]ктур[аa]\b": "Счёт-фактура",
    r"\bВс[еe]го\s*к\s*оплат[еe]\b": "Всего к оплате",
    r"\bЕдиниц[аy]\b": "Единица",
    r"\bСтавка\s+НДС\b": "Ставка НДС",
    r"\bТранспортн[ыi]е\s+услуг[иi]\b": "Транспортные услуги",
    r"Сч[еe][тt]-?ф[аa]кт[уy]ра\s*№?": "Счёт-фактура №",
    r"Иcпpaвл[еe]ни[еe]\s*№?": "Исправление №",
    r"Документ\s*об\s*отгpузк[еe]": "Документ об отгрузке",
    r"Регистрационн[ыi]й\s*номер": "Регистрационный номер",
    r"В\s*том\s*числ[еe]": "В том числе",
}

def canon_terms(s: str) -> str:
    """Приведение частых терминов к каноническому виду по regex-шаблонам."""
    out = s
    for pat, repl in TERMS_CANON.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

# ===== 3. СЛОВАРЬ КОНКРЕТНЫХ OCR-ОШИБОК =====

error_dict = {
    # стоимость / в том числе / капитал / активы / кредитные и т.п.
    "СТоиМОсти": "стоимости",
    "Стоимосtи": "стоимости",
    "стоимосtи": "стоимости",
    "стоимосt": "стоимость",
    "Тоm": "том",
    "Актиbы": "Активы",
    "КаnиТал": "Капитал",
    "п0": "по",
    "Оkуд": "ОКУД",

    # бухгалтерско-финансовая лексика
    "финансв": "финансов",
    "финаhсов": "финансов",
    "финанеовой": "финансовой",
    "бухгалтеркой": "бухгалтерской",
    "бухалтерко": "бухгалтерско",

    # прибыли и убытки
    "ибьтки": "убытки",
    "убьтки": "убытки",
    "рибьлии": "прибыли",
    "рибьль": "прибыль",

    # займы, размещенные, капитальные
    "займы Вьданне": "займы выданные",
    "вьданне": "выданные",
    "размешенные": "размещённые",
    "капиталыые": "капитальные",
    "капиталыные": "капитальные",

    # орг- и финтермы
    "организацияхй": "организациях",
    "финансовые активы,": "финансовые активы,",
    "кредитне": "кредитные",
}

def apply_error_dict(s: str) -> str:
    """Грубый словарный фикс конкретных кривых кусков."""
    out = s
    for wrong, right in error_dict.items():
        out = out.replace(wrong, right)
    return out

# ===== 4. ЧИСЛА: чистка и нормализация =====

def clean_numeric_token(tok: str) -> str:
    """
    Чистим числовой токен:
    - убираем мусорный хвост после цифр/знаков, 
      '2000,00j' -> '2000.00'
    - убираем пробелы внутри числа
    - нормализуем запятую в точку
    """
    m = re.match(r"^([0-9\s.,]+)", tok)
    if not m:
        return tok
    core = m.group(1)
    core = core.replace(" ", "").replace("\u2009", "").replace("\u00A0", "")
    core = core.replace(",", ".")
    # убираем ведущие нули
    if re.match(r"^\d+\.\d+$", core):
        int_part, frac_part = core.split(".", 1)
        int_part = int_part.lstrip("0") or "0"
        core = int_part + "." + frac_part
    else:
        core = core.lstrip("0") or "0"
    return core

def fix_numeric_tokens(s: str) -> str:
    """
    Для токенов с цифрами:
    - заменяем похожие буквы на цифры (O→0, З→3 и т.п.),
    - чистим хвосты и форматируем.
    """
    tokens = s.split(" ")
    new_tokens = []
    for tok in tokens:
        if not any(ch.isdigit() for ch in tok):
            new_tokens.append(tok)
            continue
        mapped = "".join(DIGIT_LIKE_MAP.get(ch, ch) for ch in tok)
        cleaned = clean_numeric_token(mapped)
        new_tokens.append(cleaned)
    return " ".join(new_tokens)

# ===== 5. ЛЕВЕНШТЕЙН + ЛЕКСИКОН =====

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,      # вставка
                dp[j - 1] + 1,  # удаление
                prev + cost     # замена
            )
            prev = tmp
    return dp[m]

def fuzzy_fix_word(word: str, max_dist: int = 2) -> str:
    """
    Подправляем слово по доменному словарю, 
    если оно на расстоянии ≤ max_dist от какого-то слова из DOMAIN_WORDS.
    """
    if any(ch.isdigit() for ch in word):
        return word
    if len(word) < 4:
        return word

    best = word
    best_d = max_dist + 1
    lw = word.lower()
    for w in DOMAIN_WORDS:
        d = levenshtein(lw, w.lower())
        if d < best_d:
            best_d = d
            best = w
    return best if best_d <= max_dist else word

def fuzzy_fix_line_by_lexicon(s: str) -> str:
    tokens = s.split(" ")
    fixed = [fuzzy_fix_word(tok) for tok in tokens]
    return " ".join(fixed)

# ===== 6. МЕТРИКИ "КАШИ" (используются и снаружи) =====

RUS_LETTERS = set("ёЁйцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ")
RUS_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
PUNCT = set(".,:;!?-—()[]{}\"'«»/\\")
DIGITS = set("0123456789")

def vowel_ratio(word: str) -> float:
    """Доля гласных в слове (для грубой оценки «каши»)."""
    word = ''.join(ch for ch in word if ch.isalpha())
    if not word:
        return 0.0
    vowels = sum(1 for c in word if c in RUS_VOWELS)
    return vowels / len(word)

def looks_russian_text(text: str) -> bool:
    """Есть ли вообще русские буквы."""
    return any(ch in RUS_LETTERS for ch in text)

def is_gibberish(word: str) -> bool:
    """Похоже ли слово на мусор OCR."""
    w = word.strip()
    if not w:
        return False
    if len(w) <= 2:
        return False

    # много странных символов
    weird = sum(1 for c in w if not (c.isalpha() or c.isdigit() or c in PUNCT))
    if weird / len(w) > 0.3:
        return True

    # вообще нет русских букв
    if not looks_russian_text(w) and re.search(r"[А-Яа-яЁё]", w) is None:
        return True

    # много согласных подряд
    if re.search(r"[бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ]{5,}", w):
        return True

    # очень мало гласных
    if vowel_ratio(w) < 0.15:
        return True

    return False

# ===== 7. ГЛАВНАЯ ФУНКЦИЯ НОРМАЛИЗАЦИИ СТРОКИ =====

def normalize_line(s: str) -> str:
    """
    Полный pipeline пост-обработки одной строки:
    - юникод-нормализация
    - латиница→кириллица
    - словарь конкретных ошибок
    - канонизация терминов
    - чистка чисел
    - лексикон+Левенштейн
    """
    s = nkfc(s)
    s = map_homographs(s)
    s = re.sub(r"\s+", " ", s).strip()

    s = apply_error_dict(s)
    s = canon_terms(s)
    s = fix_numeric_tokens(s)
    s = fuzzy_fix_line_by_lexicon(s)

    return s
