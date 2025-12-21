#!/usr/bin/env python3
# app.py
"""
Настольное приложение для распознавания бухгалтерских документов.
Поддерживает: УПД, ТОРГ-12, Счёт-фактура.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QTabWidget, QGroupBox, QFormLayout, QLineEdit,
    QProgressBar, QMessageBox, QSplitter, QScrollArea, QHeaderView,
    QStatusBar, QMenuBar, QAction, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / "core"))

from pipeline import DocumentPipeline, PipelineConfig, find_yolo_weights
from schemas import StructuredDocument, TableItem


class ProcessingThread(QThread):
    """Поток для обработки документа."""
    finished = pyqtSignal(object)  # StructuredDocument или Exception
    progress = pyqtSignal(str)

    def __init__(self, pipeline: DocumentPipeline, file_path: str):
        super().__init__()
        self.pipeline = pipeline
        self.file_path = file_path

    def run(self):
        try:
            self.progress.emit("Загрузка изображения...")
            results = self.pipeline.process_file(self.file_path, extract_fields=True)
            if results:
                self.finished.emit(results[0])  # Первая страница
            else:
                self.finished.emit(StructuredDocument())
        except Exception as e:
            self.finished.emit(e)


class ImagePreview(QLabel):
    """Виджет для предпросмотра изображения."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        self.setText("Перетащите файл или нажмите 'Открыть'")
        self._pixmap = None

    def set_image(self, path: str):
        """Загрузка и отображение изображения."""
        if path.lower().endswith(".pdf"):
            self.setText("PDF документ\n(предпросмотр недоступен)")
            self._pixmap = None
        else:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._pixmap = pixmap
                self._update_display()
            else:
                self.setText("Не удалось загрузить изображение")
                self._pixmap = None

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class FieldsWidget(QWidget):
    """Виджет для отображения реквизитов документа."""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Основные реквизиты
        doc_group = QGroupBox("Документ")
        doc_layout = QFormLayout(doc_group)
        self.doc_type = QLineEdit()
        # Allow editing in GUI
        self.doc_type.setReadOnly(False)
        self.doc_number = QLineEdit()
        self.doc_number.setReadOnly(False)
        self.doc_date = QLineEdit()
        self.doc_date.setReadOnly(False)
        doc_layout.addRow("Тип:", self.doc_type)
        doc_layout.addRow("Номер:", self.doc_number)
        doc_layout.addRow("Дата:", self.doc_date)
        layout.addWidget(doc_group)

        # Продавец
        seller_group = QGroupBox("Продавец")
        seller_layout = QFormLayout(seller_group)
        self.seller_name = QLineEdit()
        self.seller_name.setReadOnly(False)
        self.seller_inn = QLineEdit()
        self.seller_inn.setReadOnly(False)
        self.seller_kpp = QLineEdit()
        self.seller_kpp.setReadOnly(False)
        self.seller_address = QLineEdit()
        self.seller_address.setReadOnly(False)
        seller_layout.addRow("Наименование:", self.seller_name)
        seller_layout.addRow("ИНН:", self.seller_inn)
        seller_layout.addRow("КПП:", self.seller_kpp)
        seller_layout.addRow("Адрес:", self.seller_address)
        layout.addWidget(seller_group)

        # Покупатель
        buyer_group = QGroupBox("Покупатель")
        buyer_layout = QFormLayout(buyer_group)
        self.buyer_name = QLineEdit()
        self.buyer_name.setReadOnly(False)
        self.buyer_inn = QLineEdit()
        self.buyer_inn.setReadOnly(False)
        self.buyer_kpp = QLineEdit()
        self.buyer_kpp.setReadOnly(False)
        self.buyer_address = QLineEdit()
        self.buyer_address.setReadOnly(False)
        buyer_layout.addRow("Наименование:", self.buyer_name)
        buyer_layout.addRow("ИНН:", self.buyer_inn)
        buyer_layout.addRow("КПП:", self.buyer_kpp)
        buyer_layout.addRow("Адрес:", self.buyer_address)
        layout.addWidget(buyer_group)

        # Итоги
        totals_group = QGroupBox("Итоги")
        totals_layout = QFormLayout(totals_group)
        self.total_amount = QLineEdit()
        self.total_amount.setReadOnly(False)
        self.total_nds = QLineEdit()
        self.total_nds.setReadOnly(False)
        self.total_with_nds = QLineEdit()
        self.total_with_nds.setReadOnly(False)
        self.currency = QLineEdit()
        self.currency.setReadOnly(False)
        totals_layout.addRow("Сумма без НДС:", self.total_amount)
        totals_layout.addRow("НДС:", self.total_nds)
        totals_layout.addRow("Итого с НДС:", self.total_with_nds)
        totals_layout.addRow("Валюта:", self.currency)
        layout.addWidget(totals_group)

        layout.addStretch()

    def update_fields(self, doc: StructuredDocument):
        """Обновление полей из StructuredDocument."""
        f = doc.fields

        self.doc_type.setText(f.doc_type or "")
        self.doc_number.setText(f.doc_number or "")
        self.doc_date.setText(f.doc_date or "")

        if f.seller:
            self.seller_name.setText(f.seller.name or "")
            self.seller_inn.setText(f.seller.inn or "")
            self.seller_kpp.setText(f.seller.kpp or "")
            self.seller_address.setText(f.seller.address or "")

        if f.buyer:
            self.buyer_name.setText(f.buyer.name or "")
            self.buyer_inn.setText(f.buyer.inn or "")
            self.buyer_kpp.setText(f.buyer.kpp or "")
            self.buyer_address.setText(f.buyer.address or "")

        self.total_amount.setText(str(f.total_amount) if f.total_amount else "")
        self.total_nds.setText(str(f.total_nds) if f.total_nds else "")
        self.total_with_nds.setText(str(f.total_with_nds) if f.total_with_nds else "")
        self.currency.setText(f.currency or "")

    def clear_fields(self):
        """Очистка всех полей."""
        for widget in self.findChildren(QLineEdit):
            widget.clear()


class ItemsTableWidget(QTableWidget):
    """Таблица для отображения товарных позиций."""

    COLUMNS = [
        ("№", 40),
        ("Наименование", 250),
        ("Ед.", 60),
        ("Кол-во", 80),
        ("Цена", 100),
        ("Сумма", 100),
        ("Ставка НДС", 80),
        ("НДС", 80),
        ("Всего", 100),
    ]

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])

        header = self.horizontalHeader()
        for i, (_, width) in enumerate(self.COLUMNS):
            self.setColumnWidth(i, width)
        header.setStretchLastSection(True)

        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)

    def update_items(self, items: List[TableItem]):
        """Обновление таблицы из списка TableItem."""
        self.setRowCount(len(items))

        for row, item in enumerate(items):
            self.setItem(row, 0, QTableWidgetItem(str(item.row_num or "")))
            self.setItem(row, 1, QTableWidgetItem(item.name or ""))
            self.setItem(row, 2, QTableWidgetItem(item.unit_name or ""))
            self.setItem(row, 3, QTableWidgetItem(str(item.quantity or "")))
            self.setItem(row, 4, QTableWidgetItem(str(item.price or "")))
            self.setItem(row, 5, QTableWidgetItem(str(item.amount or "")))
            self.setItem(row, 6, QTableWidgetItem(item.nds_rate or ""))
            self.setItem(row, 7, QTableWidgetItem(str(item.nds_amount or "")))
            self.setItem(row, 8, QTableWidgetItem(str(item.total_amount or "")))

    def get_items(self) -> List[TableItem]:
        """Считывает строки из таблицы обратно в список TableItem."""
        items: List[TableItem] = []
        for row in range(self.rowCount()):
            def get(col: int) -> str:
                it = self.item(row, col)
                return it.text() if it else ""

            ti = TableItem()
            ti.row_num = int(get(0)) if get(0).isdigit() else None
            ti.name = get(1) or None
            ti.unit_name = get(2) or None
            ti.quantity = float(get(3)) if get(3).replace('.', '', 1).isdigit() else None
            ti.price = float(get(4)) if get(4).replace('.', '', 1).isdigit() else None
            ti.amount = float(get(5)) if get(5).replace('.', '', 1).isdigit() else None
            ti.nds_rate = get(6) or None
            ti.nds_amount = float(get(7)) if get(7).replace('.', '', 1).isdigit() else None
            ti.total_amount = float(get(8)) if get(8).replace('.', '', 1).isdigit() else None
            items.append(ti)
        return items


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание бухгалтерских документов")
        self.setMinimumSize(1200, 800)

        self._current_file: Optional[str] = None
        self._current_result: Optional[StructuredDocument] = None
        self._processing_thread: Optional[ProcessingThread] = None

        # Инициализация pipeline
        self._init_pipeline()
        self._init_ui()
        self._init_menu()

    def _init_pipeline(self):
        """Инициализация pipeline."""
        weights = find_yolo_weights()
        config = PipelineConfig(
            yolo_weights=weights or "",
            output_dir=str(Path(__file__).parent / "output"),
            save_artifacts=True,
        )
        self.pipeline = DocumentPipeline(config)

    def _init_ui(self):
        """Инициализация интерфейса."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("Открыть файл")
        self.btn_open.setMinimumWidth(120)
        self.btn_open.clicked.connect(self._on_open_file)

        self.btn_process = QPushButton("Распознать")
        self.btn_process.setMinimumWidth(120)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self._on_process)

        self.btn_export = QPushButton("Экспорт JSON")
        self.btn_export.setMinimumWidth(120)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_process)
        toolbar.addWidget(self.btn_export)
        toolbar.addStretch()

        # Файл label
        self.file_label = QLabel("Файл не выбран")
        toolbar.addWidget(self.file_label)

        main_layout.addLayout(toolbar)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        main_layout.addWidget(self.progress)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: Image preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_preview = ImagePreview()
        left_layout.addWidget(self.image_preview)

        splitter.addWidget(left_panel)

        # Right: Results tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: Реквизиты
        self.fields_widget = FieldsWidget()
        scroll1 = QScrollArea()
        scroll1.setWidget(self.fields_widget)
        scroll1.setWidgetResizable(True)
        self.tabs.addTab(scroll1, "Реквизиты")

        # Tab 2: Товары/услуги
        self.items_table = ItemsTableWidget()
        self.tabs.addTab(self.items_table, "Позиции")

        # Tab 3: JSON
        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.json_view, "JSON")

        # Tab 4: Предупреждения
        self.warnings_view = QTextEdit()
        self.warnings_view.setReadOnly(True)
        self.tabs.addTab(self.warnings_view, "Лог")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 700])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

    def _init_menu(self):
        """Инициализация меню."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("Файл")

        open_action = QAction("Открыть...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        export_action = QAction("Экспорт JSON...", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _on_open_file(self):
        """Открытие файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите документ",
            "",
            "Документы (*.png *.jpg *.jpeg *.pdf *.bmp *.tiff);;Все файлы (*)"
        )
        if file_path:
            self._current_file = file_path
            self._current_result = None
            self.file_label.setText(Path(file_path).name)
            self.image_preview.set_image(file_path)
            self.btn_process.setEnabled(True)
            self.btn_export.setEnabled(False)
            self.fields_widget.clear_fields()
            self.items_table.setRowCount(0)
            self.json_view.clear()
            self.warnings_view.clear()
            self.status_bar.showMessage(f"Загружен: {file_path}")

    def _on_process(self):
        """Запуск распознавания."""
        if not self._current_file:
            return

        self.btn_process.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Распознавание...")

        self._processing_thread = ProcessingThread(self.pipeline, self._current_file)
        self._processing_thread.finished.connect(self._on_processing_finished)
        self._processing_thread.progress.connect(self._on_progress)
        self._processing_thread.start()

    def _on_progress(self, message: str):
        """Обновление прогресса."""
        self.status_bar.showMessage(message)

    def _on_processing_finished(self, result):
        """Обработка результата."""
        self.progress.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)

        if isinstance(result, Exception):
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка распознавания:\n{str(result)}"
            )
            self.status_bar.showMessage("Ошибка распознавания")
            return

        self._current_result = result
        self.btn_export.setEnabled(True)

        # Обновляем UI
        self.fields_widget.update_fields(result)
        self.items_table.update_items(result.fields.items)

        # JSON
        json_str = result.model_dump_json(indent=2, ensure_ascii=False)
        self.json_view.setText(json_str)

        # Warnings/Errors
        log_text = ""
        if result.errors:
            log_text += "ОШИБКИ:\n" + "\n".join(f"  - {e}" for e in result.errors) + "\n\n"
        if result.warnings:
            log_text += "ПРЕДУПРЕЖДЕНИЯ:\n" + "\n".join(f"  - {w}" for w in result.warnings)
        self.warnings_view.setText(log_text or "Без предупреждений")

        self.status_bar.showMessage(
            f"Распознано: {result.fields.doc_type or 'Неизвестный тип'}, "
            f"позиций: {len(result.fields.items)}"
        )

    def _on_export(self):
        """Экспорт результата в JSON."""
        if not self._current_result:
            return

        default_name = Path(self._current_file).stem + "_result.json" if self._current_file else "result.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить JSON",
            default_name,
            "JSON файлы (*.json)"
        )
        if file_path:
            # Перед сохранением применяем изменения из UI к текущему результату
            try:
                # Простые поля
                f = self._current_result.fields
                f.doc_type = self.doc_type.text() or None
                f.doc_number = self.doc_number.text() or None
                f.doc_date = self.doc_date.text() or None

                f.seller.name = self.seller_name.text() or None
                f.seller.inn = self.seller_inn.text() or None
                f.seller.kpp = self.seller_kpp.text() or None
                f.seller.address = self.seller_address.text() or None

                f.buyer.name = self.buyer_name.text() or None
                f.buyer.inn = self.buyer_inn.text() or None
                f.buyer.kpp = self.buyer_kpp.text() or None
                f.buyer.address = self.buyer_address.text() or None

                f.total_amount = float(self.total_amount.text()) if self.total_amount.text() else None
                f.total_nds = float(self.total_nds.text()) if self.total_nds.text() else None
                f.total_with_nds = float(self.total_with_nds.text()) if self.total_with_nds.text() else None
                f.currency = self.currency.text() or None

                # Табличные позиции
                try:
                    f.items = self.items_table.get_items()
                except Exception:
                    pass
            except Exception:
                pass

            json_str = self._current_result.model_dump_json(indent=2, ensure_ascii=False)
            Path(file_path).write_text(json_str, encoding="utf-8")
            self.status_bar.showMessage(f"Сохранено: {file_path}")

    def _on_about(self):
        """О программе."""
        QMessageBox.about(
            self,
            "О программе",
            "Распознавание бухгалтерских документов\n\n"
            "Поддерживаемые типы:\n"
            "  - УПД (Универсальный передаточный документ)\n"
            "  - ТОРГ-12 (Товарная накладная)\n"
            "  - Счёт-фактура\n\n"
            "Технологии:\n"
            "  - PaddleOCR\n"
            "  - YOLOv8 (DocLayNet)\n"
            "  - Ollama LLM\n\n"
            "2024"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
