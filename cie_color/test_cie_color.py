from io import BytesIO
import sys
import unittest
from unittest.mock import patch

import matplotlib.image
import numpy as np

from cie_color import (
    calculate_cie_coordinates,
    create_cie_chromaticity_image,
)
from cie_color.cie_color import _chromaticity_to_srgb


class CieColorTests(unittest.TestCase):
    def test_narrow_spectrum_near_555_nm_matches_cie_1931_chromaticity(self):
        result = calculate_cie_coordinates(
            [554.0, 555.0, 556.0],
            [0.0, 1.0, 0.0],
        )

        self.assertAlmostEqual(result["cie_x"], 0.337, places=2)
        self.assertAlmostEqual(result["cie_y"], 0.659, places=2)
        self.assertAlmostEqual(
            result["cie_x"] + result["cie_y"] + result["cie_z"],
            1.0,
        )

    def test_coordinates_do_not_depend_on_intensity_scale(self):
        wavelengths = np.arange(400.0, 701.0, 5.0)
        intensity = np.exp(-((wavelengths - 555.0) / 30.0) ** 2)
        reference = calculate_cie_coordinates(wavelengths, intensity)
        scaled = calculate_cie_coordinates(wavelengths, intensity * 17.0)

        for coordinate in ("cie_x", "cie_y", "cie_z"):
            self.assertAlmostEqual(reference[coordinate], scaled[coordinate])

    def test_chromaticity_colors_are_bright_and_preserve_neutral_white(self):
        xyz = np.array(
            [
                [0.3127 / 0.329, 1.0, (1.0 - 0.3127 - 0.329) / 0.329],
                [0.64 / 0.33, 1.0, (1.0 - 0.64 - 0.33) / 0.33],
            ]
        )

        rgb = _chromaticity_to_srgb(xyz)

        np.testing.assert_allclose(rgb[0], [1.0, 1.0, 1.0], atol=1e-3)
        np.testing.assert_allclose(rgb[1], [1.0, 0.0, 0.0], atol=2e-3)

    def test_rejects_negative_intensity(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_cie_coordinates([500.0, 600.0], [1.0, -0.1])

    def test_rejects_spectrum_outside_observer_range(self):
        with self.assertRaisesRegex(ValueError, "does not overlap"):
            calculate_cie_coordinates([900.0, 910.0], [1.0, 1.0])

    def test_chromaticity_image_is_png_and_accepts_multiple_numbered_points(self):
        image = create_cie_chromaticity_image(
            [
                {"No": 1, "cie_x": 0.31, "cie_y": 0.33},
                {"No": 2, "cie_x": 0.22, "cie_y": 0.25},
            ]
        )

        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        rendered = matplotlib.image.imread(BytesIO(image))
        self.assertEqual(rendered.shape[:2], (2400, 2700))
        np.testing.assert_allclose(rendered[0, 0, :3], [1.0, 1.0, 1.0])

    def test_missing_colour_science_is_reported_when_calculation_is_requested(self):
        with patch.dict(sys.modules, {"colour": None}):
            with self.assertRaisesRegex(RuntimeError, "colour-science package"):
                calculate_cie_coordinates([500.0, 600.0], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
