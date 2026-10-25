"""CIE 1931 2 degree chromaticity calculations for emission spectra."""
from __future__ import annotations

import numpy as np
import colour


def calculate_cie_coordinates(
    wavelength_nm: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
) -> dict[str, float]:
    """Integrate an emission spectrum against the CIE 1931 2 degree observer."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    signal = np.asarray(intensity, dtype=float)
    if (
        wavelengths.ndim != 1
        or signal.ndim != 1
        or wavelengths.size != signal.size
    ):
        raise ValueError("Wavelength and intensity arrays must be one-dimensional and equal in length.")
    if wavelengths.size < 2:
        raise ValueError("At least two spectral data points are required for CIE integration.")
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(signal)):
        raise ValueError("Wavelengths and intensities must contain only finite values.")
    if np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be positive.")
    if np.any(signal < 0):
        raise ValueError(
            "CIE color coordinates require non-negative spectral intensities. "
            "Correct or remove the negative signal values first."
        )

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    signal = signal[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be unique.")

    observer = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    observer_wavelengths = np.asarray(observer.domain, dtype=float)
    matching_functions = np.asarray(observer.values, dtype=float)
    integration_min = max(float(wavelengths[0]), float(observer_wavelengths[0]))
    integration_max = min(float(wavelengths[-1]), float(observer_wavelengths[-1]))
    if integration_min > integration_max:
        raise ValueError(
            "The spectrum does not overlap the CIE observer range "
            f"({observer_wavelengths[0]:g}-{observer_wavelengths[-1]:g} nm)."
        )

    integration_wavelengths = np.unique(
        np.concatenate(
            (
                wavelengths[
                    (wavelengths >= integration_min)
                    & (wavelengths <= integration_max)
                ],
                observer_wavelengths[
                    (observer_wavelengths >= integration_min)
                    & (observer_wavelengths <= integration_max)
                ],
                [integration_min, integration_max],
            )
        )
    )
    signal_at_samples = np.interp(integration_wavelengths, wavelengths, signal)
    matching_at_samples = np.column_stack(
        [
            np.interp(
                integration_wavelengths,
                observer_wavelengths,
                matching_functions[:, index],
                left=0.0,
                right=0.0,
            )
            for index in range(3)
        ]
    )
    weighted_matching = signal_at_samples[:, np.newaxis] * matching_at_samples
    intervals = np.diff(integration_wavelengths)
    tristimulus = np.sum(
        (weighted_matching[:-1] + weighted_matching[1:])
        * intervals[:, np.newaxis]
        / 2.0,
        axis=0,
    )
    total = float(np.sum(tristimulus))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "The spectrum has no positive emission within the CIE observer range."
        )

    x, y, z = (float(value / total) for value in tristimulus)
    return {
        "cie_x": x,
        "cie_y": y,
        "cie_z": z,
        "tristimulus_X": float(tristimulus[0]),
        "tristimulus_Y": float(tristimulus[1]),
        "tristimulus_Z": float(tristimulus[2]),
        "wavelength_min_nm": integration_min,
        "wavelength_max_nm": integration_max,
    }
