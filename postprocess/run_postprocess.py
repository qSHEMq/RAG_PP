import json, os, re, sys
from normalize_text import normalize_line
from validators import valid_inn10, valid_inn12, valid_kpp, valid_ogrn, valid_ogrnip, valid_date, norm_date, norm_amount

INPUT_TXT = r"F:\AllnAll\Project_3\OCR\Output\ocr_vis_png\system_results.txt"
OUTPUT_TXT = r"F:\AllnAll\Project_3\OCR\Output\ocr_vis_png\rec_results_post.txt"

KEY_PATTERNS = [
    (re.compile(r"\bИНН[:\s]*([0-9]{10,12})"), "INN"),
    (re.compile(r"\bКПП[:\s]*([0-9]{9})"), "KPP"),
    (re.compile(r"\bОГРН[:\s]*([0-9]{13})"), "OGRN"),
    (re.compile(r"\bОГРНИП[:\s]*([0-9]{15})"), "OGRNIP"),
    (re.compile(r"\b\d{1,2}[.\s]\d{1,2}[.\s]20\d{2}\b"), "DATE"),
    (re.compile(r"\b\d{1,3}(\s?\d{3})*(,\d{2})?\b"), "AMOUNT"),
]

def fix_field(tag, text):
    if tag=="INN":
        num = re.search(r"\d{10,12}", text).group()
        return num if (valid_inn10(num) or valid_inn12(num)) else num  # можно прикрутить правку 1 символа
    if tag=="KPP":
        num = re.search(r"\d{9}", text).group()
        return num if valid_kpp(num) else num
    if tag=="OGRN":
        num = re.search(r"\d{13}", text).group()
        return num if valid_ogrn(num) else num
    if tag=="OGRNIP":
        num = re.search(r"\d{15}", text).group()
        return num if valid_ogrnip(num) else num
    if tag=="DATE":
        return norm_date(text)
    if tag=="AMOUNT":
        return norm_amount(text)
    return text

def process_line(s):
    s2 = normalize_line(s)
    # точечная валидация ключевых полей
    for pat, tag in KEY_PATTERNS:
        for m in pat.finditer(s2):
            fixed = fix_field(tag, m.group())
            s2 = s2.replace(m.group(), fixed)
    return s2

def main():
    if not os.path.exists(INPUT_TXT):
        print("Не найден rec_results.txt. Сначала запусти инференс с --draw_img_save_dir.")
        sys.exit(1)
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    out = [process_line(l) for l in lines]
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for l in out: f.write(l + "\n")
    print("Готово:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
