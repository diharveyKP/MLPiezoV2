"""
Model catalog discovery for GeoStudio model sets.

This module scans a user-supplied root path, discovers model files
(`.gsz` archives and extracted `.xml` models), and emits a normalized
Section -> Model -> Analysis catalog with phreatic surface and geometry
metadata.
"""

from __future__ import annotations

import json
import re
from xml.sax.saxutils import escape
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


@dataclass
class PointRecord:
    point_id: str
    x: float
    y: float


@dataclass
class PhreaticSurfaceRecord:
    source_kind: str
    data_point_count: int
    point_refs: List[str] = field(default_factory=list)
    points: List[PointRecord] = field(default_factory=list)


@dataclass
class GeometryRecord:
    point_count: int
    x_min: Optional[float]
    x_max: Optional[float]
    y_min: Optional[float]
    y_max: Optional[float]
    envelope_points: List[PointRecord] = field(default_factory=list)


@dataclass
class AnalysisRecord:
    analysis_id: str
    analysis_name: str
    analysis_kind: Optional[str]
    analysis_method: Optional[str]
    xml_source: str
    phreatic_surface: PhreaticSurfaceRecord
    geometry: GeometryRecord


@dataclass
class ModelRecord:
    section_name: str
    model_name: str
    source_type: str
    source_path: str
    title: Optional[str]
    author: Optional[str]
    date: Optional[str]
    analyses: List[AnalysisRecord] = field(default_factory=list)


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned.lower() or "item"


def sanitize_sheet_name(value: str, used_names: Optional[set[str]] = None) -> str:
    cleaned = re.sub(r"[\[\]\*\?/\\:]", "_", value).strip()
    cleaned = cleaned or "Sheet"
    cleaned = cleaned[:31]

    if used_names is None:
        return cleaned

    candidate = cleaned
    counter = 2
    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{cleaned[: max(0, 31 - len(suffix))]}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _text(element: Optional[ET.Element], tag: str) -> Optional[str]:
    if element is None:
        return None
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _find_points_by_id(root: ET.Element) -> Dict[str, PointRecord]:
    points: Dict[str, PointRecord] = {}

    for point in root.findall(".//Geometries/Geometry/Points/Point"):
        point_id = point.get("ID")
        if not point_id:
            continue
        try:
            points[point_id] = PointRecord(
                point_id=point_id,
                x=float(point.get("X", "0")),
                y=float(point.get("Y", "0")),
            )
        except ValueError:
            continue

    for datapoint in root.findall(".//StabilityItems/StabilityItem/Entry/DataPoints/DataPoint"):
        point_id = datapoint.get("Number")
        if not point_id or point_id in points:
            continue
        try:
            points[point_id] = PointRecord(
                point_id=point_id,
                x=float(datapoint.get("X", "0")),
                y=float(datapoint.get("Y", "0")),
            )
        except ValueError:
            continue

    return points


def _extract_top_envelope(points: Dict[str, PointRecord]) -> List[PointRecord]:
    if not points:
        return []

    arr = sorted([(p.x, p.y) for p in points.values()], key=lambda item: item[0])
    x_values = [point[0] for point in arr]
    y_values = [point[1] for point in arr]

    if len(arr) <= 50:
        envelope = arr
    else:
        x_min = min(x_values)
        x_max = max(x_values)
        if x_min == x_max:
            envelope = arr
        else:
            step = (x_max - x_min) / 50
            bins = [x_min + step * i for i in range(51)]
            sampled: List[Tuple[float, float]] = []
            for i in range(len(bins) - 1):
                left = bins[i]
                right = bins[i + 1]
                if i == len(bins) - 2:
                    bucket = [point for point in arr if left <= point[0] <= right]
                else:
                    bucket = [point for point in arr if left <= point[0] < right]
                if not bucket:
                    continue
                sampled.append(max(bucket, key=lambda item: item[1]))
            envelope = sampled if sampled else arr

    return [
        PointRecord(point_id=str(i + 1), x=float(x), y=float(y))
        for i, (x, y) in enumerate(envelope)
    ]


def _extract_geometry(root: ET.Element) -> GeometryRecord:
    points = _find_points_by_id(root)
    if not points:
        return GeometryRecord(
            point_count=0,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            envelope_points=[],
        )

    return GeometryRecord(
        point_count=len(points),
        x_min=min(point.x for point in points.values()),
        x_max=max(point.x for point in points.values()),
        y_min=min(point.y for point in points.values()),
        y_max=max(point.y for point in points.values()),
        envelope_points=_extract_top_envelope(points),
    )


def _extract_phreatic_surface(root: ET.Element) -> PhreaticSurfaceRecord:
    points_by_id = _find_points_by_id(root)
    stability_entry = root.find(".//StabilityItems/StabilityItem/Entry")

    piezo_refs = []
    if stability_entry is not None:
        piezo_surface = stability_entry.find("./PiezometricSurfaces/PiezometricSurface")
        piezo_data_points = []
        if piezo_surface is not None:
            piezo_data_points = piezo_surface.findall("./DataPoints/DataPoint")
        piezo_refs = [
            (dp.text or "").strip()
            for dp in piezo_data_points
            if (dp.text or "").strip()
        ]

    if piezo_refs:
        points = [points_by_id[ref] for ref in piezo_refs if ref in points_by_id]
        return PhreaticSurfaceRecord(
            source_kind="piezometric_surface_refs",
            data_point_count=len(points),
            point_refs=piezo_refs,
            points=points,
        )

    explicit = []
    datapoint_elements = []
    if stability_entry is not None:
        datapoint_elements = stability_entry.findall("./DataPoints/DataPoint")

    for datapoint in datapoint_elements:
        number = datapoint.get("Number") or str(len(explicit) + 1)
        try:
            explicit.append(
                PointRecord(
                    point_id=number,
                    x=float(datapoint.get("X", "0")),
                    y=float(datapoint.get("Y", "0")),
                )
            )
        except ValueError:
            continue

    return PhreaticSurfaceRecord(
        source_kind="explicit_datapoints",
        data_point_count=len(explicit),
        point_refs=[point.point_id for point in explicit],
        points=explicit,
    )


def _build_analysis_record(root: ET.Element, xml_source: str, analysis_element: Optional[ET.Element] = None) -> AnalysisRecord:
    analysis_id = _text(analysis_element, "ID") or "1"
    analysis_name = _text(analysis_element, "Name") or Path(xml_source).stem
    analysis_kind = _text(analysis_element, "Kind")
    analysis_method = _text(analysis_element, "Method")

    return AnalysisRecord(
        analysis_id=analysis_id,
        analysis_name=analysis_name,
        analysis_kind=analysis_kind,
        analysis_method=analysis_method,
        xml_source=xml_source,
        phreatic_surface=_extract_phreatic_surface(root),
        geometry=_extract_geometry(root),
    )


def _parse_xml_bytes(xml_bytes: bytes) -> ET.Element:
    return ET.fromstring(xml_bytes)


def _candidate_model_xmls(root_path: Path) -> List[Path]:
    xml_files = sorted(root_path.rglob("*.xml"))
    model_xmls: List[Path] = []

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue

        analyses = root.findall("./Analyses/Analysis")
        if not analyses:
            continue

        analysis_names = {_text(analysis, "Name") for analysis in analyses}
        sibling_dirs = {path.name for path in xml_path.parent.iterdir() if path.is_dir()}
        if analysis_names.intersection(sibling_dirs) or len(analyses) > 1:
            model_xmls.append(xml_path)

    return model_xmls


def _derive_section_name(root_path: Path, model_path: Path) -> str:
    rel_parts = model_path.relative_to(root_path).parts
    if len(rel_parts) >= 2:
        return rel_parts[0]
    return root_path.name


def _discover_from_xml_file(root_path: Path, xml_path: Path) -> ModelRecord:
    root = ET.parse(xml_path).getroot()
    file_info = root.find("./FileInfo")
    analyses = root.findall("./Analyses/Analysis")

    model = ModelRecord(
        section_name=_derive_section_name(root_path, xml_path),
        model_name=xml_path.stem,
        source_type="xml",
        source_path=str(xml_path),
        title=file_info.get("Title") if file_info is not None else None,
        author=file_info.get("Author") if file_info is not None else None,
        date=file_info.get("Date") if file_info is not None else None,
        analyses=[],
    )

    for analysis in analyses:
        analysis_name = _text(analysis, "Name") or f"Analysis_{len(model.analyses) + 1}"
        analysis_xml = xml_path.parent / analysis_name / xml_path.name
        if analysis_xml.exists():
            analysis_root = ET.parse(analysis_xml).getroot()
            source = str(analysis_xml)
        else:
            analysis_root = root
            source = str(xml_path)
        model.analyses.append(_build_analysis_record(analysis_root, source, analysis))

    return model


def _discover_from_gsz(root_path: Path, archive_path: Path) -> ModelRecord:
    with zipfile.ZipFile(archive_path, "r") as zf:
        xml_entries = [info for info in zf.infolist() if info.filename.lower().endswith(".xml")]
        top_level = next((info for info in xml_entries if "/" not in info.filename.strip("/")), None)
        if top_level is None:
            raise ValueError(f"No top-level XML found in {archive_path}")

        root = _parse_xml_bytes(zf.read(top_level))
        file_info = root.find("./FileInfo")
        analyses = root.findall("./Analyses/Analysis")

        model = ModelRecord(
            section_name=_derive_section_name(root_path, archive_path),
            model_name=archive_path.stem,
            source_type="gsz",
            source_path=str(archive_path),
            title=file_info.get("Title") if file_info is not None else None,
            author=file_info.get("Author") if file_info is not None else None,
            date=file_info.get("Date") if file_info is not None else None,
            analyses=[],
        )

        for analysis in analyses:
            analysis_name = _text(analysis, "Name") or f"Analysis_{len(model.analyses) + 1}"
            expected_entry = f"{analysis_name}/{top_level.filename}"
            if expected_entry in zf.namelist():
                analysis_root = _parse_xml_bytes(zf.read(expected_entry))
                source = f"{archive_path}!/{expected_entry}"
            else:
                analysis_root = root
                source = f"{archive_path}!/{top_level.filename}"
            model.analyses.append(_build_analysis_record(analysis_root, source, analysis))

    return model


def discover_models(root_path: str | Path) -> List[ModelRecord]:
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    model_records: List[ModelRecord] = []

    for archive_path in sorted(root.rglob("*.gsz")):
        model_records.append(_discover_from_gsz(root, archive_path))

    xml_candidates = _candidate_model_xmls(root)
    archive_xml_parents = {Path(record.source_path).parent for record in model_records if record.source_type == "gsz"}
    for xml_path in xml_candidates:
        if xml_path.parent in archive_xml_parents:
            continue
        model_records.append(_discover_from_xml_file(root, xml_path))

    model_records.sort(key=lambda record: (record.section_name.lower(), record.model_name.lower()))
    return model_records


def _analysis_rows(models: Sequence[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        for analysis in model.analyses:
            rows.append(
                {
                    "section_name": model.section_name,
                    "model_name": model.model_name,
                    "analysis_id": analysis.analysis_id,
                    "analysis_name": analysis.analysis_name,
                    "analysis_kind": analysis.analysis_kind,
                    "analysis_method": analysis.analysis_method,
                    "source_type": model.source_type,
                    "source_path": model.source_path,
                    "analysis_xml_source": analysis.xml_source,
                    "title": model.title,
                    "author": model.author,
                    "date": model.date,
                    "phreatic_source_kind": analysis.phreatic_surface.source_kind,
                    "phreatic_point_count": analysis.phreatic_surface.data_point_count,
                    "geometry_point_count": analysis.geometry.point_count,
                    "geometry_x_min": analysis.geometry.x_min,
                    "geometry_x_max": analysis.geometry.x_max,
                    "geometry_y_min": analysis.geometry.y_min,
                    "geometry_y_max": analysis.geometry.y_max,
                }
            )
    return rows


def _phreatic_rows(models: Sequence[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        for analysis in model.analyses:
            for idx, point in enumerate(analysis.phreatic_surface.points, start=1):
                rows.append(
                    {
                        "section_name": model.section_name,
                        "model_name": model.model_name,
                        "analysis_name": analysis.analysis_name,
                        "analysis_id": analysis.analysis_id,
                        "point_index": idx,
                        "point_id": point.point_id,
                        "x": point.x,
                        "y": point.y,
                        "source_kind": analysis.phreatic_surface.source_kind,
                    }
                )
    return rows


def _geometry_rows(models: Sequence[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        for analysis in model.analyses:
            for idx, point in enumerate(analysis.geometry.envelope_points, start=1):
                rows.append(
                    {
                        "section_name": model.section_name,
                        "model_name": model.model_name,
                        "analysis_name": analysis.analysis_name,
                        "analysis_id": analysis.analysis_id,
                        "point_index": idx,
                        "point_id": point.point_id,
                        "x": point.x,
                        "y": point.y,
                    }
                )
    return rows


def _column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xml_cell(value: object) -> str:
    if value is None:
        return "<c/>"
    if isinstance(value, bool):
        return f'<c t="inlineStr"><is><t>{str(value).lower()}</t></is></c>'
    if isinstance(value, (int, float)):
        return f"<c><v>{value}</v></c>"
    return f'<c t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _worksheet_xml(rows: List[Dict[str, object]]) -> str:
    if not rows:
        rows = [{"message": "No records"}]

    headers = list(rows[0].keys())
    xml_rows = []

    header_cells = "".join(
        f'<c r="{_column_name(i)}1" t="inlineStr"><is><t>{escape(header)}</t></is></c>'
        for i, header in enumerate(headers, start=1)
    )
    xml_rows.append(f'<row r="1">{header_cells}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells = []
        for col_index, header in enumerate(headers, start=1):
            cell_ref = f"{_column_name(col_index)}{row_index}"
            cell_xml = _xml_cell(row.get(header))
            if cell_xml == "<c/>":
                cells.append(f'<c r="{cell_ref}"/>')
            else:
                cells.append(cell_xml.replace("<c", f'<c r="{cell_ref}"', 1))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    dimension = f"A1:{_column_name(len(headers))}{len(rows) + 1}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<dimension ref=\"{dimension}\"/>"
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def _write_simple_xlsx(workbook_path: Path, sheets: List[Tuple[str, List[Dict[str, object]]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    for idx, (sheet_name, _) in enumerate(sheets, start=1):
        workbook_xml.append(f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    workbook_xml.extend(["</sheets>", "</workbook>"])

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheets) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    with zipfile.ZipFile(workbook_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", "".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows))


def _analysis_input_sheets(models: Sequence[ModelRecord]) -> List[Tuple[str, List[Dict[str, object]]]]:
    sheets: List[Tuple[str, List[Dict[str, object]]]] = []
    used_names: set[str] = set()

    for model in models:
        for analysis in model.analyses:
            base_name = f"{model.section_name}_{analysis.analysis_id}"
            if analysis.analysis_name:
                base_name = f"{base_name}_{analysis.analysis_name}"
            sheet_name = sanitize_sheet_name(base_name, used_names)

            rows = [
                {
                    "section_name": model.section_name,
                    "model_name": model.model_name,
                    "analysis_id": analysis.analysis_id,
                    "analysis_name": analysis.analysis_name,
                    "point_index": idx,
                    "point_id": point.point_id,
                    "x": point.x,
                    "y": point.y,
                }
                for idx, point in enumerate(analysis.phreatic_surface.points, start=1)
            ]

            if not rows:
                rows = [
                    {
                        "section_name": model.section_name,
                        "model_name": model.model_name,
                        "analysis_id": analysis.analysis_id,
                        "analysis_name": analysis.analysis_name,
                        "point_index": None,
                        "point_id": None,
                        "x": None,
                        "y": None,
                    }
                ]

            sheets.append((sheet_name, rows))

    return sheets


def export_catalog(models: Sequence[ModelRecord], output_dir: str | Path) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configs_dir = output_path / "model_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    catalog_json = output_path / "model_catalog.json"
    workbook_path = output_path / "model_catalog.xlsx"
    inputs_workbook_path = output_path / "phreatic_surface_inputs.xlsx"

    catalog_payload = {"models": [asdict(model) for model in models]}
    catalog_json.write_text(json.dumps(catalog_payload, indent=2), encoding="utf-8")

    for model in models:
        config_payload = {
            "section_name": model.section_name,
            "model_name": model.model_name,
            "source_type": model.source_type,
            "source_path": model.source_path,
            "title": model.title,
            "author": model.author,
            "date": model.date,
            "analyses": [asdict(analysis) for analysis in model.analyses],
        }
        file_name = f"{sanitize_slug(model.section_name)}__{sanitize_slug(model.model_name)}.json"
        (configs_dir / file_name).write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    _write_simple_xlsx(
        workbook_path,
        [
            ("analyses", _analysis_rows(models)),
            ("phreatic_surfaces", _phreatic_rows(models)),
            ("geometry_envelope", _geometry_rows(models)),
        ],
    )
    _write_simple_xlsx(inputs_workbook_path, _analysis_input_sheets(models))

    return {
        "catalog_json": catalog_json,
        "workbook": workbook_path,
        "inputs_workbook": inputs_workbook_path,
        "configs_dir": configs_dir,
    }
