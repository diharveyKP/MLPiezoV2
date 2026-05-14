import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipelines"))

from ensemble import FOSEnsemble
from geostudio_interface import GeoStudioInterface, PhreaticSurfaceData
from model_workflow import _extract_full_gsz_model

with patch("sys.stdout", new=io.StringIO()):
    from pipeline_generate_dataset import generate_valid_sample
    from train_models import analyze_confidence_intervals


class ConstantModel:
    def __init__(self, value: float, name: str):
        self.value = value
        self.name = name

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class SequencedSampler:
    def __init__(self, samples):
        self.samples = [np.array(sample, dtype=float) for sample in samples]
        self.index = 0

    def sample_random(self, n, seed=None):
        del n, seed
        sample = self.samples[self.index]
        self.index += 1
        return np.array([sample], dtype=float)


class DummyEnsemble:
    def predict_with_uncertainty(self, X):
        n = len(X)
        return (
            np.full(n, 1.2),
            np.full(n, 0.05),
            np.full(n, 0.9),
        )

    def predict_quantiles(self, X, quantiles):
        n = len(X)
        result = {}
        for q in quantiles:
            if q < 0.5:
                result[q] = np.full(n, 1.1)
            else:
                result[q] = np.full(n, 1.3)
        return result


class RegressionTests(unittest.TestCase):
    def test_archive_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            archive_path = temp_root / "bad.gsz"
            destination = temp_root / "extract"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("model.xml", "<Root />")
                zf.writestr("../outside.txt", "owned")

            with self.assertRaises(ValueError):
                _extract_full_gsz_model(archive_path, destination)

            self.assertFalse((temp_root / "outside.txt").exists())

    def test_run_analysis_restores_xml_and_ignores_stale_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_path = root / "template.xml"
            analysis_dir = root / "Analysis" / "001"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            csv_path = analysis_dir / "slip_surface.csv"
            geocmd_exe = root / "geocmd.exe"
            geocmd_exe.write_text("", encoding="utf-8")

            original_xml = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<Root>"
                "<StabilityItems><StabilityItem><Entry>"
                '<DataPoints Len="1"><DataPoint Number="1" X="0.0" Y="0.0" /></DataPoints>'
                "<PiezometricSurfaces><PiezometricSurface>"
                '<DataPoints Len="1"><DataPoint>1</DataPoint></DataPoints>'
                "</PiezometricSurface></PiezometricSurfaces>"
                "</Entry></StabilityItem></StabilityItems>"
                "</Root>"
            )
            xml_path.write_text(original_xml, encoding="utf-8")

            pd.DataFrame({"SlipFOS": [9.99] * 50, "Other": list(range(50))}).to_csv(csv_path, index=False)

            interface = GeoStudioInterface(str(xml_path), str(geocmd_exe), "SlipFOS")
            phreatic_data = PhreaticSurfaceData(
                control_point_numbers=[1, 2, 3],
                control_point_x=[0.0, 1.0, 2.0],
                control_point_y=[10.0, 9.0, 8.0],
                interpolated_x=np.array([0.0, 1.0, 2.0]),
                interpolated_y=np.array([10.0, 9.0, 8.0]),
            )

            def fake_run_geocmd(*, xml_path, geocmd_exe, retries=3, timeout=300):
                del geocmd_exe, retries, timeout
                mutated = Path(xml_path).read_text(encoding="utf-8")
                self.assertIn('Y="10.00"', mutated)
                pd.DataFrame({"SlipFOS": [1.23] * 50, "Other": list(range(50))}).to_csv(csv_path, index=False)
                return True

            with patch("geostudio_interface.run_geocmd", side_effect=fake_run_geocmd):
                result = interface.run_analysis(phreatic_data, sample_id=1)

            self.assertTrue(result.success)
            self.assertEqual(result.fos, 1.23)
            self.assertEqual(xml_path.read_text(encoding="utf-8"), original_xml)

    def test_generate_valid_sample_revalidates_after_sync(self):
        sampler = SequencedSampler([
            [3.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ])

        sample, attempts = generate_valid_sample(
            sampler=sampler,
            control_x=np.array([0.0, 1.0, 2.0]),
            max_gradient=10.0,
            sync_groups=[[0, 2]],
            max_attempts=5,
        )

        np.testing.assert_allclose(sample, np.array([3.0, 3.0, 3.0]))
        self.assertEqual(attempts, 2)

    def test_analyze_confidence_intervals_no_name_error(self):
        ensemble = DummyEnsemble()
        X_test = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_test = np.array([0.9, 1.1, 1.3, 1.5])

        with patch("sys.stdout", new=io.StringIO()):
            analyze_confidence_intervals(ensemble, X_test, y_test)

    def test_weighted_uncertainty_uses_ensemble_weights(self):
        ensemble = FOSEnsemble([ConstantModel(1.0, "a"), ConstantModel(3.0, "b")])
        ensemble.is_fitted = True
        ensemble.model_weights = np.array([0.25, 0.75])

        X = np.array([[0.0], [1.0]])
        pred = ensemble.predict(X)
        mean, std, _ = ensemble.predict_with_uncertainty(X)

        np.testing.assert_allclose(pred, np.array([2.5, 2.5]))
        np.testing.assert_allclose(mean, pred)
        np.testing.assert_allclose(std, np.array([np.sqrt(0.75), np.sqrt(0.75)]))


if __name__ == "__main__":
    unittest.main()
