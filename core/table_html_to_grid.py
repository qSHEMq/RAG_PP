# table_html_to_grid.py
from typing import List
from bs4 import BeautifulSoup

def html_to_grid(html: str) -> List[List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    grid: List[List[str]] = []
    for tr in table.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            row.append(cell.get_text(" ", strip=True))
        if row:
            grid.append(row)
    return grid
