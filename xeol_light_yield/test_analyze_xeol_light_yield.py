import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyze_xeol_light_yield import (
    calculate_absolute_light_yield,
    calculate_deposited_xray_energy,
    calculate_detected_photon_rate,
    correct_collection_efficiency,
    correct_spectral_response,
    generate_xeol_report,
    main,
    read_spectral_response,
    subtract_background,
)


class XeolLightYieldTests(unittest.TestCase):
    def test_response_correction_uses_integration_time_and_nm_grid(self):
        result = correct_spectral_response(
            [(400, 20), (500, 20)],
            [(400, 2e-9), (500, 2e-9)],
            integration_time_s=2,
        )
        self.assertEqual(result, [(400.0, 2e-8), (500.0, 2e-8)])

    def test_background_is_interpolated_and_subtracted(self):
        corrected = subtract_background(
            [(400, 12), (500, 18)],
            [(400, 2), (450, 5), (500, 8)],
        )
        self.assertEqual(corrected, [(400.0, 10.0), (500.0, 10.0)])

    def test_collection_efficiency_is_corrected_per_wavelength(self):
        corrected = correct_collection_efficiency(
            [(400, 10), (500, 20)],
            [(400, 0.5), (500, 0.25)],
        )
        self.assertEqual(corrected, [(400.0, 20.0), (500.0, 80.0)])

    def test_yield_and_deposited_energy_use_actual_exposure(self):
        deposited = calculate_deposited_xray_energy(100, 5, 8.1, 0.5)
        self.assertEqual(deposited, 2025)
        result = calculate_absolute_light_yield(
            [(400, 1e-12), (600, 1e-12)],
            exposure_time_s=5,
            collection_efficiency=0.5,
            deposited_energy_kev=deposited,
        )
        self.assertAlmostEqual(
            result["emitted_photon_rate_s-1"],
            calculate_detected_photon_rate([(400, 1e-12), (600, 1e-12)]) * 2,
        )
        self.assertAlmostEqual(
            result["light_yield_photons_per_keV"],
            result["emitted_photons"] / 2025,
        )
        longer_exposure_deposited = calculate_deposited_xray_energy(100, 50, 8.1, 0.5)
        longer_exposure_result = calculate_absolute_light_yield(
            [(400, 1e-12), (600, 1e-12)],
            exposure_time_s=50,
            collection_efficiency=0.5,
            deposited_energy_kev=longer_exposure_deposited,
        )
        self.assertAlmostEqual(
            result["light_yield_photons_per_keV"],
            longer_exposure_result["light_yield_photons_per_keV"],
        )

    def test_response_table_requires_explicit_units_and_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            table_path = Path(temp_dir) / "response.csv"
            table_path.write_text(
                "wavelength_nm,watts_per_nm_per_count_rate\n400,1e-9\n500,2e-9\n",
                encoding="utf-8",
            )
            self.assertEqual(read_spectral_response(table_path)[0], (400.0, 1e-9))
            with self.assertRaisesRegex(ValueError, "does not cover"):
                correct_spectral_response([(390, 2), (500, 2)], read_spectral_response(table_path), 1)
            table_path.write_text("wavelength_nm,response\n400,1\n500,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must have columns"):
                read_spectral_response(table_path)

    def test_report_records_range_and_uncertainty_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spectrum_path = root / "sample.txt"
            response_path = root / "response.csv"
            output_path = root / "report.json"
            spectrum_path.write_text("400 10\n500 20\n", encoding="utf-8")
            response_path.write_text(
                "wavelength_nm,watts_per_nm_per_count_rate\n400,1e-9\n500,1e-9\n",
                encoding="utf-8",
            )
            inputs = {name: None for name in (
                "absolute spectral response",
                "collection efficiency",
                "incident flux",
                "X-ray energy",
                "deposited fraction",
            )}
            report = generate_xeol_report(
                spectrum_path,
                response_path,
                output_path,
                {
                    "composition": "test",
                    "thickness_mm": 0.5,
                    "collection_geometry": "test",
                    "radiometric_calibration_identifier": "response-test",
                    "wavelength_calibration_identifier": "wavelength-test",
                },
                {"estimated_deposited_energy_keV": 1},
                {"light_yield_photons_per_keV": 2},
                inputs,
                acquisition_metadata={"signal_basis": "counts_per_second"},
            )
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["measured_wavelength_range_nm"], [400.0, 500.0])
            self.assertEqual(
                saved_report["calibration"]["radiometric_calibration_identifier"],
                "response-test",
            )
            self.assertEqual(
                saved_report["calibration"]["wavelength_calibration_identifier"],
                "wavelength-test",
            )
            self.assertIsNone(
                saved_report["uncertainty"][
                    "light_yield_standard_uncertainty_photons_per_keV"
                ]
            )
            self.assertEqual(saved_report["acquisition"]["signal_basis"], "counts_per_second")

    def test_cli_writes_absolute_yield_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spectrum_path = root / "sample.txt"
            response_path = root / "response.csv"
            output_path = root / "report.json"
            spectrum_path.write_text("400 100\n600 100\n", encoding="utf-8")
            response_path.write_text(
                "wavelength_nm,watts_per_nm_per_count_rate\n400,1e-9\n600,1e-9\n",
                encoding="utf-8",
            )
            arguments = [
                "analyze_xeol_light_yield.py",
                "--spectrum", str(spectrum_path),
                "--response", str(response_path),
                "--output", str(output_path),
                "--collection-efficiency", "0.02",
                "--integration-time-s", "1",
                "--exposure-time-s", "10",
                "--incident-flux-photons-s", "100",
                "--xray-energy-kev", "8.1",
                "--deposited-fraction", "0.5",
                "--sample-composition", "test sample",
                "--sample-thickness-mm", "0.5",
                "--collection-geometry", "test geometry",
                "--radiometric-calibration-identifier", "test response",
                "--wavelength-calibration-identifier", "test wavelength certificate",
            ]
            with patch("sys.argv", arguments), patch("sys.stdout", new_callable=io.StringIO):
                main()
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertGreater(
                report["results"]["light_yield_photons_per_keV"],
                0,
            )
            self.assertEqual(report["xray"]["estimated_deposited_energy_keV"], 4050)
            self.assertEqual(
                report["calibration"]["wavelength_calibration_identifier"],
                "test wavelength certificate",
            )

    def test_invalid_deposition_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Deposited fraction"):
            calculate_deposited_xray_energy(100, 1, 8.1, 0)


if __name__ == "__main__":
    unittest.main()
