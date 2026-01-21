# core/schemas.py
"""Расширенные схемы для извлечения данных из бухгалтерских документов."""
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class PartyInfo(BaseModel):
    """Информация о контрагенте (продавец/покупатель)."""
    name: Optional[str] = Field(None, description="Наименование организации")
    inn: Optional[str] = Field(None, description="ИНН (10 или 12 цифр)")
    kpp: Optional[str] = Field(None, description="КПП (9 цифр)")
    address: Optional[str] = Field(None, description="Юридический адрес")
    bank_account: Optional[str] = Field(None, description="Расчётный счёт")
    bank_name: Optional[str] = Field(None, description="Наименование банка")
    bik: Optional[str] = Field(None, description="БИК банка")
    corr_account: Optional[str] = Field(None, description="Корр. счёт")


class TableItem(BaseModel):
    """Строка товарной таблицы."""
    row_num: Optional[int] = Field(None, description="№ п/п")
    name: Optional[str] = Field(None, description="Наименование товара/услуги")
    unit_code: Optional[str] = Field(None, description="Код единицы измерения")
    unit_name: Optional[str] = Field(None, description="Единица измерения")
    quantity: Optional[float] = Field(None, description="Количество")
    price: Optional[float] = Field(None, description="Цена за единицу")
    amount: Optional[float] = Field(None, description="Сумма без НДС")
    nds_rate: Optional[str] = Field(None, description="Ставка НДС (20%, 10%, без НДС)")
    nds_amount: Optional[float] = Field(None, description="Сумма НДС")
    total_amount: Optional[float] = Field(None, description="Сумма с НДС")


class DocumentFields(BaseModel):
    """Полный набор реквизитов документа."""
    # Тип и идентификация
    doc_type: Optional[str] = Field(None, description="Тип документа: УПД, ТОРГ-12, Счёт-фактура")
    doc_number: Optional[str] = Field(None, description="Номер документа")
    doc_date: Optional[str] = Field(None, description="Дата документа (ДД.ММ.ГГГГ)")

    # Для счёт-фактуры
    sf_number: Optional[str] = Field(None, description="Номер счёт-фактуры")
    sf_date: Optional[str] = Field(None, description="Дата счёт-фактуры")
    correction_number: Optional[str] = Field(None, description="Номер исправления")
    correction_date: Optional[str] = Field(None, description="Дата исправления")

    # Контрагенты
    seller: Optional[PartyInfo] = Field(default_factory=PartyInfo, description="Продавец")
    buyer: Optional[PartyInfo] = Field(default_factory=PartyInfo, description="Покупатель")
    consignor: Optional[str] = Field(None, description="Грузоотправитель")
    consignee: Optional[str] = Field(None, description="Грузополучатель")

    # Договор/основание
    contract_number: Optional[str] = Field(None, description="Номер договора")
    contract_date: Optional[str] = Field(None, description="Дата договора")

    # Итоги
    total_quantity: Optional[float] = Field(None, description="Всего количество")
    total_amount: Optional[float] = Field(None, description="Итого сумма без НДС")
    total_nds: Optional[float] = Field(None, description="Итого НДС")
    total_with_nds: Optional[float] = Field(None, description="Всего к оплате с НДС")
    currency: Optional[str] = Field(None, description="Валюта (руб., USD, EUR)")

    # Позиции таблицы
    items: List[TableItem] = Field(default_factory=list, description="Товарные позиции")


class StructuredDocument(BaseModel):
    """Результат распознавания документа."""
    fields: DocumentFields = Field(default_factory=DocumentFields)
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Уверенность распознавания")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения")
    errors: List[str] = Field(default_factory=list, description="Ошибки")
