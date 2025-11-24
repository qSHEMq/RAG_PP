from gpt4all import GPT4All

model_path = r"C:\Users\fifik\AppData\Local\nomic.ai\GPT4All\qwen2-1_5b-instruct-q4_0.gguf"
model = GPT4All(model_path)

def llm_fix_word(model, word: str) -> str:
    prompt = (
        "Ты исправляешь слово после OCR в русском бухгалтерском документе.\n"
        "Дано одно слово с ошибками. Нужно вернуть только исправленное слово на кириллице.\n"
        "Не добавляй новых слов, не объясняй, не комментируй.\n"
        f"Слово: {word}\n"
        "Исправленное слово:"
    )
    out = model.generate(prompt, max_tokens=10, temp=0.1)
    # берём первую непустую строку
    cand = out.strip().splitlines()[0].strip()
    # если модель начала 'думать' — откатываемся
    if " " in cand or len(cand) < 3 or len(cand) > len(word) + 5:
        return word
    return cand

with model.chat_session():
    w1 = normalize_mixed_word("Cm0имOсть")
    print("После нормализации:", w1)
    w2 = llm_fix_word(model, w1)
    print("После Qwen:", w2)
