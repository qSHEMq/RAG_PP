#!/usr/bin/env python3
"""
CLI для системы распознавания бухгалтерских документов.

Использование:
    python cli.py --input "./file.pdf" --output "./result.json"
    python cli.py --input "./documents/" --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core"))

from pipeline import DocumentPipeline, PipelineConfig, find_yolo_weights


def main():
    parser = argparse.ArgumentParser(
        description="Система распознавания бухгалтерских документов (УПД, ТОРГ-12, Счёт-фактура)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python cli.py --input doc.png --output result.json
  python cli.py --input ./documents/ --batch
  python cli.py --input doc.pdf --llm --model qwen2.5:7b-instruct
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Путь к файлу или папке (PNG, JPG, PDF)")
    parser.add_argument("--output", "-o", help="Путь для сохранения JSON (по умолчанию: auto)")
    parser.add_argument("--batch", action="store_true", help="Пакетная обработка папки")
    parser.add_argument("--llm", action="store_true",
                        help="Включить LLM-извлечение полей (рекомендуется модель 7b+)")
    parser.add_argument("--model", default="qwen2.5:7b-instruct",
                        help="Модель Ollama для LLM (рекомендуется 7b+: qwen2.5:7b-instruct, llama3.1:8b)")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Хост Ollama")
    parser.add_argument("--yolo-conf", type=float, default=0.10, help="YOLO confidence threshold")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: путь не найден: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Конфигурация
    weights = find_yolo_weights()
    if not weights and args.verbose:
        print("Предупреждение: веса YOLO не найдены, детекция таблиц отключена")

    config = PipelineConfig(
        yolo_weights=weights or "",
        output_dir="output",
        save_artifacts=True,
        yolo_conf=args.yolo_conf,
        ollama_model=args.model,
        ollama_host=args.host,
    )

    pipeline = DocumentPipeline(config)

    # Пакетная обработка
    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            print(f"Ошибка: --batch требует папку, получен файл: {input_path}", file=sys.stderr)
            sys.exit(1)

        files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + \
                list(input_path.glob("*.jpeg")) + list(input_path.glob("*.pdf"))

        if not files:
            print(f"Ошибка: файлы не найдены в {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Пакетная обработка: {len(files)} файлов")
        results = pipeline.process_batch([str(f) for f in files], extract_fields=args.llm)

        success_count = 0
        error_count = 0
        for file_path, docs in results:
            has_error = any(doc.errors for doc in docs)
            if has_error:
                error_count += 1
                if args.verbose:
                    print(f"  ОШИБКА: {file_path}")
            else:
                success_count += 1
                if args.verbose:
                    print(f"  OK: {file_path}")

        print(f"\nГотово: {success_count} успешно, {error_count} с ошибками")
        sys.exit(0 if error_count == 0 else 1)

    # Обработка одного файла
    if args.verbose:
        print(f"Обработка: {input_path}")
        print(f"Веса YOLO: {weights or 'НЕТ'}")
        print(f"Модель LLM: {args.model}")
        print("-" * 50)

    try:
        results = pipeline.process_file(str(input_path), extract_fields=args.llm)
    except Exception as e:
        print(f"Ошибка обработки: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Определяем путь для вывода
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / input_path.stem / f"{input_path.stem}_structured.json"

    # Формируем итоговый JSON
    if len(results) == 1:
        output_data = results[0].model_dump()
    else:
        output_data = {"pages": [doc.model_dump() for doc in results]}

    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Вывод результатов
    for i, doc in enumerate(results):
        if args.verbose:
            print(f"\n=== Страница {i + 1} ===")

        if doc.errors:
            print("ОШИБКИ:", file=sys.stderr)
            for err in doc.errors:
                print(f"  - {err}", file=sys.stderr)

        if doc.warnings and args.verbose:
            print("Предупреждения:")
            for warn in doc.warnings:
                print(f"  - {warn}")

        f = doc.fields
        if args.verbose:
            print(f"\nТип документа: {f.doc_type or 'Не определён'}")
            print(f"Номер: {f.doc_number or '-'}")
            print(f"Дата: {f.doc_date or '-'}")

            if f.seller:
                print(f"\nПродавец: {f.seller.name or '-'}")
                print(f"  ИНН: {f.seller.inn or '-'}, КПП: {f.seller.kpp or '-'}")

            if f.buyer:
                print(f"\nПокупатель: {f.buyer.name or '-'}")
                print(f"  ИНН: {f.buyer.inn or '-'}, КПП: {f.buyer.kpp or '-'}")

            print(f"\nИтого без НДС: {f.total_amount or '-'}")
            print(f"НДС: {f.total_nds or '-'}")
            print(f"Всего с НДС: {f.total_with_nds or '-'}")

            if f.items:
                print(f"\nПозиций: {len(f.items)}")
                for item in f.items[:5]:
                    print(f"  {item.row_num or '-'}. {item.name or '-'}: {item.quantity or '-'} x {item.price or '-'}")
                if len(f.items) > 5:
                    print(f"  ... и ещё {len(f.items) - 5} позиций")

    print(f"\nРезультат: {output_path}")

    # Возвращаем код ошибки если были ошибки
    has_errors = any(doc.errors for doc in results)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
