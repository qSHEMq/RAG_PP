# postprocess/llm_postprocess.py
from __future__ import annotations

import json
from pathlib import Path

from llm_structure import structure_with_ollama

def main():
    # 1) найди последний raw.json или укажи конкретный путь
    # проще — укажи конкретный raw:
    raw_path = Path("output\Screenshot_1\Screenshot_1_raw.json")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    structured = structure_with_ollama(
        raw,
        model="qwen2.5:0.5b-instruct",
        host="http://127.0.0.1:11434",
    )

    out_path = raw_path.with_name(raw_path.name.replace("_raw.json", "_structured.json"))
    out_path.write_text(structured.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ Done:", out_path)

if __name__ == "__main__":
    main()