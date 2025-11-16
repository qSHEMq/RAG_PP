import argparse
import os
from gpt4all import GPT4All

from normalize_text import is_gibberish, vowel_ratio 

MODEL_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"

SYSTEM_PROMPT = (
    "Ты действуешь как корректор ошибок OCR в бухгалтерских документах.\n"
    "Тебе даётся короткий фрагмент текста с ошибками распознавания.\n"
    "Твоя задача: максимально восстановить нормальный русский текст,\n"
    "сохранить смысл, не добавлять нового смысла и не сокращать фразу.\n"
    "Не пиши ничего кроме исправленного текста.\n"
)

def need_llm_fix(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # отбрасываем совсем короткое и очень длинное
    if len(s) < 10:
        return False
    if len(s) > 200:
        return False

    tokens = [t for t in s.split() if t]
    if not tokens:
        return False

    # посчитаем долю "мусорных" слов
    gib = 0
    long_tokens = 0
    for t in tokens:
        if len(t) >= 4:
            long_tokens += 1
            if is_gibberish(t):
                gib += 1

    if long_tokens == 0:
        return False

    ratio = gib / long_tokens

    # если хотя бы 1 "мусорное" слово среди длинных — уже повод чинить
    if ratio >= 0.15:
        return True

    # запасной критерий: среднее соотношение гласных в длинных словах
    vr_vals = [vowel_ratio(t) for t in tokens if len(t) >= 5]
    if vr_vals:
        avg_vr = sum(vr_vals) / len(vr_vals)
        # если в среднем мало гласных → текст похож на кашу
        if avg_vr < 0.25:
            return True

    return False


def correct_line_with_llm(model: GPT4All, line: str) -> str:
    prompt = (
        SYSTEM_PROMPT
        + "\nВходной текст (как распознал OCR):\n"
        + line.strip()
        + "\n\nИсправленный текст:\n"
    )
    out = model.generate(prompt, max_tokens=80, temp=0.1)
    return out.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="входной txt после постобработки (rec_results_post)")
    parser.add_argument("--output", required=True, help="выходной txt с ЛЛМ-коррекцией")
    parser.add_argument("--model", default=MODEL_NAME, help="имя/файл модели GPT4All (.gguf)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Не найден входной файл:", args.input)
        return

    # загружаем модель (первый запуск может долго качать .gguf)
    print("Загружаю модель:", args.model)
    model = GPT4All(args.model)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    total = len(lines)
    fixed = 0

    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        if need_llm_fix(s):
            try:
                new_s = correct_line_with_llm(model, s)
                fixed += 1
                print(f"[{i+1}/{total}] FIX:", s, "=>", new_s)
                out_lines.append(new_s + "\n")
            except Exception as e:
                print(f"[{i+1}/{total}] ERROR LLM:", e)
                out_lines.append(s + "\n")
        else:
            out_lines.append(s + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Готово. Строк всего: {total}, через ЛЛМ пропущено: {fixed}")
    print("Результат сохранён в:", args.output)

if __name__ == "__main__":
    main()
