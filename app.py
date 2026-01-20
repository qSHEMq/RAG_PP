#!/usr/bin/env python3
"""
Веб-интерфейс для распознавания бухгалтерских документов на Gradio.
Поддерживает: УПД, ТОРГ-12, Счёт-фактура.
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Отключаем proxy чтобы избежать ошибки httpx с socks://
# ValueError: Unknown scheme for proxy URL
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

import gradio as gr

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Глобальный pipeline (инициализируется лениво)
_pipeline = None


def get_pipeline():
    """Получение или создание pipeline (ленивая загрузка)."""
    global _pipeline
    if _pipeline is None:
        # Импортируем только при первом использовании
        from pipeline import DocumentPipeline, PipelineConfig, find_yolo_weights

        weights = find_yolo_weights()
        config = PipelineConfig(
            yolo_weights=weights or "",
            output_dir=str(Path(__file__).parent / "output"),
            save_artifacts=True,
        )
        _pipeline = DocumentPipeline(config)
    return _pipeline


def _get_file_path(file_input: Any) -> Optional[str]:
    """Нормализация значения входного файла из Gradio."""
    if not file_input:
        return None
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict) and "path" in file_input:
        return file_input.get("path")
    if isinstance(file_input, (list, tuple)) and file_input:
        return _get_file_path(file_input[0])
    if hasattr(file_input, "name"):
        return file_input.name
    return None


def _build_preview_html(file_input: Any) -> str:
    """Возвращает HTML для предпросмотра PDF/изображения."""
    file_path = _get_file_path(file_input)
    if not file_path:
        return "<div>Загрузите PDF или изображение для предпросмотра.</div>"
    path = Path(file_path)
    if not path.exists():
        return "<div>Файл не найден.</div>"

    mime, _ = mimetypes.guess_type(path.name)
    mime = (mime or "").lower()
    try:
        data = path.read_bytes()
    except Exception:
        return "<div>Не удалось прочитать файл.</div>"

    b64 = base64.b64encode(data).decode("ascii")
    if path.suffix.lower() == ".pdf" or mime == "application/pdf":
        data_url = f"data:application/pdf;base64,{b64}"
        return (
            "<div style=\"position:relative;width:100%;height:420px;\">"
            "<button type=\"button\" data-fullscreen data-target=\".preview-frame\" "
            "style=\"position:absolute;top:8px;right:8px;z-index:2;\">Во весь экран</button>"
            "<iframe "
            "class=\"preview-frame\" "
            f"src=\"{data_url}\" "
            "style=\"width:100%;height:100%;border:1px solid #ddd;border-radius:8px;\" "
            "allowfullscreen "
            "allow=\"fullscreen\" "
            "></iframe>"
            "</div>"
        )
    if mime.startswith("image/"):
        data_url = f"data:{mime};base64,{b64}"
        return (
            "<div style=\"position:relative;width:100%;height:420px;\">"
            "<button type=\"button\" data-fullscreen data-target=\".preview-image\" "
            "style=\"position:absolute;top:8px;right:8px;z-index:2;\">Во весь экран</button>"
            f"<img src=\"{data_url}\" class=\"preview-image\" "
            "style=\"width:100%;height:100%;border:1px solid #ddd;border-radius:8px;object-fit:contain;display:block;\"/>"
            "</div>"
        )
    return "<div>Неподдерживаемый формат файла.</div>"


def process_document(
    file_input: Any,
    use_llm: bool = False,
) -> Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str]:
    """
    Обработка документа и возврат результатов.

    Returns:
        Кортеж значений для всех полей интерфейса
    """
    file_path = _get_file_path(file_input)
    if not file_path:
        return ("", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Загрузите PDF или изображение")

    try:
        # Ленивая загрузка pipeline
        pipeline = get_pipeline()
        results = pipeline.process_file(file_path, extract_fields=use_llm)

        if not results:
            return ("", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Не удалось обработать документ")

        doc = results[0]
        f = doc.fields

        # Формируем JSON
        json_str = doc.model_dump_json(indent=2, ensure_ascii=False)

        # Формируем лог
        log_lines = []
        if doc.errors:
            log_lines.append("ОШИБКИ:")
            log_lines.extend(f"  - {e}" for e in doc.errors)
        if doc.warnings:
            log_lines.append("ПРЕДУПРЕЖДЕНИЯ:")
            log_lines.extend(f"  - {w}" for w in doc.warnings)
        log_text = "\n".join(log_lines) if log_lines else "Без предупреждений"

        # Формируем таблицу позиций
        items_md = ""
        if f.items:
            items_md = "| № | Наименование | Ед. | Кол-во | Цена | Сумма | НДС | Всего |\n"
            items_md += "|---|-------------|-----|--------|------|-------|-----|-------|\n"
            for item in f.items:
                items_md += f"| {item.row_num or ''} | {item.name or ''} | {item.unit_name or ''} | {item.quantity or ''} | {item.price or ''} | {item.amount or ''} | {item.nds_rate or ''} | {item.total_amount or ''} |\n"

        return (
            f.doc_type or "",
            f.doc_number or "",
            f.doc_date or "",
            f.seller.name if f.seller else "",
            f.seller.inn if f.seller else "",
            f.seller.kpp if f.seller else "",
            f.seller.address if f.seller else "",
            f.buyer.name if f.buyer else "",
            f.buyer.inn if f.buyer else "",
            f.buyer.kpp if f.buyer else "",
            f.buyer.address if f.buyer else "",
            str(f.total_amount) if f.total_amount else "",
            str(f.total_nds) if f.total_nds else "",
            str(f.total_with_nds) if f.total_with_nds else "",
            items_md,
            json_str,
            log_text,
        )

    except Exception as e:
        return ("", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", f"Ошибка: {str(e)}")


def export_json(json_text: str) -> Optional[str]:
    """Экспорт JSON в файл."""
    if not json_text:
        return None

    # Создаём временный файл
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    tmp.write(json_text)
    tmp.close()
    return tmp.name


def create_interface() -> gr.Blocks:
    """Создание интерфейса Gradio."""

    with gr.Blocks(title="Распознавание бухгалтерских документов") as app:

        gr.Markdown("# Распознавание бухгалтерских документов")
        gr.Markdown("Загрузите PDF или изображение документа (УПД, ТОРГ-12, Счёт-фактура) для распознавания.")

        with gr.Row():
            with gr.Column(scale=1):
                # Загрузка PDF/изображения
                file_input = gr.File(
                    label="Документ (PDF/PNG/JPG)",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    file_count="single",
                    type="filepath",
                )
                gr.Markdown("### Предпросмотр")
                preview_html = gr.HTML(
                    js_on_load="""
                        element.addEventListener('click', (e) => {
                            const btn = e.target.closest('[data-fullscreen]');
                            if (!btn) return;
                            const selector = btn.getAttribute('data-target');
                            const target = selector ? element.querySelector(selector) : null;
                            if (!target || !target.requestFullscreen) return;
                            target.requestFullscreen().catch(() => {});
                        });
                    """
                )

                with gr.Row():
                    use_llm = gr.Checkbox(
                        label="Использовать LLM",
                        value=False,
                        info="Включить LLM-извлечение (требуется Ollama, модель 7b+)"
                    )

                process_btn = gr.Button("Распознать", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Реквизиты"):
                        with gr.Group():
                            gr.Markdown("### Документ")
                            with gr.Row():
                                doc_type = gr.Textbox(label="Тип", interactive=True)
                                doc_number = gr.Textbox(label="Номер", interactive=True)
                                doc_date = gr.Textbox(label="Дата", interactive=True)

                        with gr.Group():
                            gr.Markdown("### Продавец")
                            seller_name = gr.Textbox(label="Наименование", interactive=True)
                            with gr.Row():
                                seller_inn = gr.Textbox(label="ИНН", interactive=True)
                                seller_kpp = gr.Textbox(label="КПП", interactive=True)
                            seller_address = gr.Textbox(label="Адрес", interactive=True)

                        with gr.Group():
                            gr.Markdown("### Покупатель")
                            buyer_name = gr.Textbox(label="Наименование", interactive=True)
                            with gr.Row():
                                buyer_inn = gr.Textbox(label="ИНН", interactive=True)
                                buyer_kpp = gr.Textbox(label="КПП", interactive=True)
                            buyer_address = gr.Textbox(label="Адрес", interactive=True)

                        with gr.Group():
                            gr.Markdown("### Итоги")
                            with gr.Row():
                                total_amount = gr.Textbox(label="Сумма без НДС", interactive=True)
                                total_nds = gr.Textbox(label="НДС", interactive=True)
                                total_with_nds = gr.Textbox(label="Итого с НДС", interactive=True)

                    with gr.TabItem("Позиции"):
                        items_table = gr.Markdown(label="Товары/услуги")

                    with gr.TabItem("JSON"):
                        json_output = gr.Code(label="JSON", language="json", interactive=True)
                        export_btn = gr.Button("Скачать JSON")
                        json_file = gr.File(label="Скачать", visible=False)

                    with gr.TabItem("Лог"):
                        log_output = gr.Textbox(label="Предупреждения и ошибки", lines=10)

        # Обработчики
        outputs = [
            doc_type, doc_number, doc_date,
            seller_name, seller_inn, seller_kpp, seller_address,
            buyer_name, buyer_inn, buyer_kpp, buyer_address,
            total_amount, total_nds, total_with_nds,
            items_table, json_output, log_output,
        ]

        process_btn.click(
            fn=process_document,
            inputs=[file_input, use_llm],
            outputs=outputs,
        )

        file_input.change(
            fn=_build_preview_html,
            inputs=[file_input],
            outputs=[preview_html],
        )

        export_btn.click(
            fn=export_json,
            inputs=[json_output],
            outputs=[json_file],
        )

        # Информация о тестовых документах
        gr.Markdown("---")
        gr.Markdown("""
### Тестовые документы
Для тестирования используйте файлы из папок:
- `data/TORG-12/Screenshot_1.png`
- `data/TORG-12/screen5.pdf`
- `data/УПД/Screenshot_1.png`
        """)

    return app


def main():
    """Запуск приложения."""
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
