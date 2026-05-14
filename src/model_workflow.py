"""
Workbook-driven model preparation and execution.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from xlsx_io import read_workbook


@dataclass
class AnalysisInput:
    section_name: str
    model_name: str
    analysis_id: str
    analysis_name: str
    points: List[Tuple[str, float, float]]


@dataclass
class PreparedModel:
    section_name: str
    model_name: str
    source_type: str
    source_path: Path
    prepared_dir: Path
    model_xml_path: Path
    analyses: List[AnalysisInput] = field(default_factory=list)
    analysis_xml_paths: Dict[str, Path] = field(default_factory=dict)
    result_csv_paths: Dict[str, Path] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    prepared: PreparedModel
    success: bool
    exit_code: Optional[int]
    fos_by_analysis: Dict[str, Optional[float]]
    error: Optional[str]


_PRINT_LOCK = Lock()


def _safe_print(message: str = "") -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in value).strip("_") or "item"


def _load_catalog(catalog_path: str | Path) -> Dict[tuple[str, str], Dict[str, object]]:
    payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    mapping: Dict[tuple[str, str], Dict[str, object]] = {}
    for model in payload.get("models", []):
        key = (model["section_name"], model["model_name"])
        mapping[key] = model
    return mapping


def load_analysis_inputs(workbook_path: str | Path) -> List[AnalysisInput]:
    sheets = read_workbook(workbook_path)
    inputs: List[AnalysisInput] = []

    for sheet in sheets:
        if not sheet.rows:
            continue
        first = sheet.rows[0]
        points: List[Tuple[str, float, float]] = []

        for row in sheet.rows:
            point_id = row.get("point_id", "")
            x_raw = row.get("x", "")
            y_raw = row.get("y", "")
            if not point_id or x_raw == "" or y_raw == "":
                continue
            points.append((point_id, float(x_raw), float(y_raw)))

        if first.get("section_name") and first.get("model_name") and first.get("analysis_name") and points:
            inputs.append(
                AnalysisInput(
                    section_name=first["section_name"],
                    model_name=first["model_name"],
                    analysis_id=first.get("analysis_id", ""),
                    analysis_name=first["analysis_name"],
                    points=points,
                )
            )

    return inputs


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _handle_remove_readonly(func, path, exc_info) -> None:
    try:
        os.chmod(path, 0o700)
        func(path)
    except OSError:
        raise exc_info[1]


def _group_inputs_by_model(inputs: Sequence[AnalysisInput]) -> Dict[tuple[str, str], List[AnalysisInput]]:
    grouped: Dict[tuple[str, str], List[AnalysisInput]] = {}
    for item in inputs:
        grouped.setdefault((item.section_name, item.model_name), []).append(item)
    return grouped


def _short_model_dir(root: Path, model_index: int, section_name: str, model_name: str) -> Path:
    return root / f"m{model_index:03d}_{_safe_name(section_name)[:12]}_{_safe_name(model_name)[:24]}"


def _extract_full_gsz_model(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    top_level_xml: Optional[Path] = None

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            filename = info.filename.strip("/")
            if not filename:
                continue
            target = destination / Path(*filename.split("/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            if "/" not in filename and target.suffix.lower() == ".xml":
                top_level_xml = target

    if top_level_xml is None:
        raise FileNotFoundError(f"No top-level XML found in {archive_path}")
    return top_level_xml


def _copy_full_xml_model(source_xml: Path, destination: Path) -> Path:
    source_dir = source_xml.parent
    _copy_tree(source_dir, destination)
    copied_xml = destination / source_xml.name
    if not copied_xml.exists():
        raise FileNotFoundError(f"Top-level XML not copied for {source_xml}")
    return copied_xml


def _update_datapoints(xml_path: Path, points: Sequence[Tuple[str, float, float]]) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    entry = root.find(".//StabilityItems/StabilityItem/Entry")
    if entry is None:
        raise ValueError(f"Stability entry not found in {xml_path}")

    data_points = entry.find("./DataPoints")
    if data_points is None:
        raise ValueError(f"DataPoints not found in {xml_path}")

    data_points.clear()
    data_points.set("Len", str(len(points)))
    for point_id, x, y in points:
        elem = ET.SubElement(data_points, "DataPoint")
        elem.set("Number", str(point_id))
        elem.set("X", f"{x:.5f}")
        elem.set("Y", f"{y:.5f}")

    piezo_surface = entry.find("./PiezometricSurfaces/PiezometricSurface/DataPoints")
    if piezo_surface is not None:
        piezo_surface.clear()
        piezo_surface.set("Len", str(len(points)))
        for point_id, _, _ in points:
            elem = ET.SubElement(piezo_surface, "DataPoint")
            elem.text = str(point_id)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def prepare_models_from_workbook(
    catalog_path: str | Path,
    workbook_path: str | Path,
    working_root: str | Path,
) -> List[PreparedModel]:
    catalog = _load_catalog(catalog_path)
    inputs = load_analysis_inputs(workbook_path)
    grouped_inputs = _group_inputs_by_model(inputs)
    working_root_path = Path(working_root).resolve()
    prepared_models: List[PreparedModel] = []

    for model_index, ((section_name, model_name), model_inputs) in enumerate(grouped_inputs.items(), start=1):
        catalog_model = catalog.get((section_name, model_name))
        if catalog_model is None:
            raise KeyError(f"Model not found in catalog: {(section_name, model_name)}")

        destination = _short_model_dir(working_root_path, model_index, section_name, model_name)
        source_type = str(catalog_model["source_type"])
        source_path = Path(str(catalog_model["source_path"]))

        _safe_print(f"[PREP MODEL {model_index}] {section_name} | {model_name}")
        _safe_print(f"    Source: {source_path} ({source_type})")
        _safe_print(f"    Working dir: {destination}")

        if destination.exists():
            shutil.rmtree(destination, onexc=_handle_remove_readonly)

        if source_type == "gsz":
            model_xml_path = _extract_full_gsz_model(source_path, destination)
        else:
            model_xml_path = _copy_full_xml_model(source_path, destination)

        prepared = PreparedModel(
            section_name=section_name,
            model_name=model_name,
            source_type=source_type,
            source_path=source_path,
            prepared_dir=destination,
            model_xml_path=model_xml_path,
            analyses=model_inputs,
        )

        model_xml_name = model_xml_path.name
        for analysis in model_inputs:
            analysis_xml = destination / analysis.analysis_name / model_xml_name
            if not analysis_xml.exists():
                raise FileNotFoundError(
                    f"Analysis XML not found for {section_name} | {model_name} | {analysis.analysis_name}: {analysis_xml}"
                )
            _update_datapoints(analysis_xml, analysis.points)
            prepared.analysis_xml_paths[analysis.analysis_name] = analysis_xml
            prepared.result_csv_paths[analysis.analysis_name] = destination / analysis.analysis_name / "001" / "slip_surface.csv"
            _safe_print(f"    Updated {analysis.analysis_name}: {len(analysis.points)} phreatic points")

        prepared_models.append(prepared)

    return prepared_models


def _read_fos_from_csv(csv_path: Path, column_name: str = "SlipFOS") -> Optional[float]:
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        values: List[float] = []
        for row in reader:
            raw = row.get(column_name)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return min(values) if values else None


def _run_single_prepared_model(
    item: PreparedModel,
    geocmd: Path,
    timeout_seconds: Optional[int],
    index: int,
    total: int,
) -> ExecutionResult:
    command = [str(geocmd), str(item.model_xml_path), "/solve"]
    _safe_print("\n" + "=" * 70)
    _safe_print(f"[RUN MODEL {index}/{total}] {item.section_name} | {item.model_name}")
    _safe_print(f"XML: {item.model_xml_path}")
    _safe_print(f"Analyses: {len(item.analyses)}")
    _safe_print(f"CMD: {' '.join(command)}")
    _safe_print("=" * 70)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(item.prepared_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        _safe_print(f"[FAILED MODEL {index}/{total}] could not start GeoCMD: {exc}")
        return ExecutionResult(item, False, None, {}, str(exc))

    line_queue: queue.Queue[Optional[str]] = queue.Queue()

    def _reader() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                line_queue.put(line.rstrip())
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    output_lines: List[str] = []
    start_time = time.time()
    stream_closed = False

    while True:
        elapsed = time.time() - start_time
        if timeout_seconds is not None and elapsed > timeout_seconds:
            process.kill()
            reader_thread.join(timeout=2)
            _safe_print(f"[TIMEOUT MODEL {index}/{total}] exceeded {timeout_seconds} seconds")
            return ExecutionResult(item, False, None, {}, f"Timed out after {timeout_seconds} seconds")

        try:
            line = line_queue.get(timeout=0.5)
        except queue.Empty:
            if process.poll() is not None and stream_closed:
                break
            continue

        if line is None:
            stream_closed = True
            if process.poll() is not None:
                break
            continue

        text = line.rstrip()
        output_lines.append(text)
        _safe_print(f"[MODEL {index}/{total}] {text}")

        if process.poll() is not None and stream_closed:
            break

    exit_code = process.wait(timeout=5)

    fos_by_analysis = {
        analysis.analysis_name: _read_fos_from_csv(item.result_csv_paths[analysis.analysis_name])
        for analysis in item.analyses
    }
    success = exit_code == 0
    error = None if success else ("\n".join(output_lines) or f"Exit code {exit_code}")[:4000]
    status = "SUCCESS" if success else "FAILED"
    fos_preview = ", ".join(f"{name}={value}" for name, value in list(fos_by_analysis.items())[:3])
    _safe_print(f"[{status} MODEL {index}/{total}] exit_code={exit_code} {fos_preview}")
    return ExecutionResult(item, success, exit_code, fos_by_analysis, error)


def run_prepared_models(
    prepared: Sequence[PreparedModel],
    geocmd_path: str | Path,
    timeout_seconds: Optional[int] = None,
    max_parallel: int = 1,
) -> List[ExecutionResult]:
    geocmd = Path(geocmd_path)
    if not geocmd.exists():
        raise FileNotFoundError(f"GeoCMD not found: {geocmd}")

    total = len(prepared)
    if total == 0:
        return []

    worker_count = max(1, min(max_parallel, total))
    _safe_print(f"\nRunning with up to {worker_count} parallel model process(es)")

    indexed_items = list(enumerate(prepared, start=1))
    results_by_index: Dict[int, ExecutionResult] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_run_single_prepared_model, item, geocmd, timeout_seconds, index, total): index
            for index, item in indexed_items
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results_by_index[index] = future.result()

    return [results_by_index[index] for index, _ in indexed_items]


def write_run_summary(results: Sequence[ExecutionResult], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for result in results:
        base = {
            "section_name": result.prepared.section_name,
            "model_name": result.prepared.model_name,
            "model_xml_path": str(result.prepared.model_xml_path),
            "success": result.success,
            "exit_code": result.exit_code,
            "error": result.error,
        }
        if result.fos_by_analysis:
            for analysis_name, fos in result.fos_by_analysis.items():
                row = dict(base)
                row["analysis_name"] = analysis_name
                row["fos"] = fos
                rows.append(row)
        else:
            row = dict(base)
            row["analysis_name"] = ""
            row["fos"] = None
            rows.append(row)

    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["section_name"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return output
