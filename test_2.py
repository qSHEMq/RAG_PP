import unicodedata

# Таблица похожих латинских → кириллица
LATIN_TO_CYR = {
    "A": "А", "a": "а",
    "B": "В", "E": "Е", "e": "е",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "H": "Н", "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "C": "С", "c": "с",
    "T": "Т", "X": "Х",
    "Y": "У", "y": "у",
    "N": "Н", "n": "п",  # на вкус, можно убрать
    "r": "г",
    "u": "и",
    "w": "ш",
    "m": "т", "t": "т"

}

def normalize_mixed_word(word: str) -> str:
    result = []
    for ch in word:
        # если цифра внутри слова, часто это «0» вместо «о» или «1» вместо «l/и»
        if ch.isdigit():
            if ch == "0":
                result.append("о")
            elif ch == "1":
                result.append("и")
            else:
                # можно либо игнорировать, либо оставлять
                continue
        else:
            # пробуем через таблицу
            if ch in LATIN_TO_CYR:
                result.append(LATIN_TO_CYR[ch])
            else:
                result.append(ch)
    return "".join(result)

print(normalize_mixed_word("Cm0имOсть"))   # ожидаем что-то близкое к "Смоиность"/"Смоимость"
print(normalize_mixed_word("СТоиМосtи"))   # "СТоиМОсти"
print(normalize_mixed_word("ОбязаtелсТо")) # "ОбязателсТо" → потом уже лексикой/LLM добиваем