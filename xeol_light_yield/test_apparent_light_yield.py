import math
import unittest

from apparent_light_yield import (
    calculate_absorption_corrected_light_yield,
    calculate_apparent_light_yield,
    calculate_monoenergetic_absorption_fraction,
    calculate_scintillator_equivalent_thickness,
)


class ApparentLightYieldTests(unittest.TestCase):
    def test_apparent_yield_uses_common_range_and_reference_scale(self):
        result = calculate_apparent_light_yield(
            [(400, 2), (600, 2)],
            [(500, 1), (700, 1)],
            reference_light_yield_photons_per_mev=30000,
        )
        self.assertEqual(result["wavelength_min_nm"], 500.0)
        self.assertEqual(result["wavelength_max_nm"], 600.0)
        self.assertEqual(result["sample_integrated_signal"], 200.0)
        self.assertEqual(result["reference_integrated_signal"], 100.0)
        self.assertEqual(result["apparent_light_yield_photons_per_mev"], 60000.0)

    def test_detector_correction_is_applied_to_both_spectra(self):
        result = calculate_apparent_light_yield(
            [(400, 1), (500, 2), (600, 1)],
            [(400, 1), (500, 1), (600, 2)],
            reference_light_yield_photons_per_mev=10000,
            detector_correction=[(400, 1), (500, 2), (600, 1)],
        )
        self.assertAlmostEqual(
            result["apparent_light_yield_photons_per_mev"],
            10000 * 500 / 350,
        )

    def test_correction_requires_full_common_wavelength_coverage(self):
        with self.assertRaisesRegex(ValueError, "must cover"):
            calculate_apparent_light_yield(
                [(400, 1), (600, 1)],
                [(400, 1), (600, 1)],
                30000,
                detector_correction=[(450, 1), (550, 1)],
            )

    def test_absorption_correction_normalizes_both_materials(self):
        corrected = calculate_absorption_corrected_light_yield(
            40000,
            sample_absorption_fraction=0.5,
            reference_absorption_fraction=0.25,
        )
        self.assertEqual(corrected, 20000)

    def test_monochromatic_absorption_fraction_uses_mm_to_cm_conversion(self):
        fraction = calculate_monoenergetic_absorption_fraction(2, 5)
        self.assertAlmostEqual(fraction, 1 - math.exp(-1))

    def test_scintillator_equivalent_thickness_matches_si_equation(self):
        thickness = calculate_scintillator_equivalent_thickness(
            total_thickness_mm=0.5,
            scintillator_mass_fraction=0.5,
            scintillator_density_g_cm3=5,
            matrix_density_g_cm3=1,
        )
        self.assertAlmostEqual(thickness, 0.5 / 6)

    def test_invalid_yield_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            calculate_apparent_light_yield(
                [(400, 0), (500, 0)],
                [(400, 1), (500, 1)],
                30000,
            )
        with self.assertRaisesRegex(ValueError, "at most 1"):
            calculate_absorption_corrected_light_yield(30000, 1.1, 0.5)


if __name__ == "__main__":
    unittest.main()
