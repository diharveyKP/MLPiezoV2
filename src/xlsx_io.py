"""
Minimal XLSX reader for workbook-driven GeoStudio workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re
import zipfile
import xml.etree.ElementTree as ET


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class WorksheetData:
    name: str
    rows: List[Dict[str, str]]


def _col_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    col = match.group(1)
    index = 0
    for char in col:
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: List[str] = []
    for item in root.findall("main:si", NS):
        texts = [node.text or "" for node in item.findall(".//main:t", NS)]
        strings.append("".join(texts))
    return strings


def _sheet_map(zf: zipfile.ZipFile) -> List[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pkg:Relationship", NS)
    }

    sheets: List[tuple[str, str]] = []
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[f"{{{NS['rel']}}}id"]
        target = rel_targets[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((name, target))
    return sheets


def _cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//main:t", NS)]
        return "".join(texts)

    value_node = cell.find("main:v", NS)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return ""
    return value


def _parse_sheet(xml_bytes: bytes, shared_strings: List[str]) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    rows = root.findall("main:sheetData/main:row", NS)
    if not rows:
        return []

    parsed_rows: List[Dict[int, str]] = []
    max_col = 0

    for row in rows:
        parsed: Dict[int, str] = {}
        for cell in row.findall("main:c", NS):
            cell_ref = cell.attrib.get("r", "")
            col_index = _col_to_index(cell_ref)
            parsed[col_index] = _cell_value(cell, shared_strings)
            max_col = max(max_col, col_index)
        parsed_rows.append(parsed)

    headers = [parsed_rows[0].get(i, "").strip() for i in range(max_col + 1)]
    data_rows: List[Dict[str, str]] = []

    for parsed in parsed_rows[1:]:
        row_dict: Dict[str, str] = {}
        for i, header in enumerate(headers):
            if not header:
                continue
            row_dict[header] = parsed.get(i, "").strip()
        if any(value != "" for value in row_dict.values()):
            data_rows.append(row_dict)

    return data_rows


def read_workbook(path: str | Path) -> List[WorksheetData]:
    workbook_path = Path(path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    with zipfile.ZipFile(workbook_path, "r") as zf:
        shared_strings = _read_shared_strings(zf)
        sheets = _sheet_map(zf)
        return [
            WorksheetData(name=name, rows=_parse_sheet(zf.read(target), shared_strings))
            for name, target in sheets
        ]
