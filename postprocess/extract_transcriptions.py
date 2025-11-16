import re
import argparse
from difflib import SequenceMatcher

TARGET = "transcription"

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_transcriptions(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ищем пары "ключ": "значение"
    pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
    pairs = pattern.findall(text)

    results = []

    for key, value in pairs:
        # сравниваем ключ и "transcription"
        if similar(key.lower(), TARGET) > 0.6:  # 60% похожести
            results.append(value.strip())

    print(f"Найдено transcription-полей: {len(results)}")

    with open(output_path, "w", encoding="utf-8") as out:
        for line in results:
            out.write(line + "\n")

    print("Готово, сохранено в:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    extract_transcriptions(args.input, args.output)
