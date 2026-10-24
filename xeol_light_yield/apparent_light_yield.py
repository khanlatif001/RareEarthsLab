"""Reference-based apparent light-yield calculations for XEOL spectra."""
from __future__ import annotations

import math
from typing import Sequence


Spectrum = list[tuple[float, float]]


def _validate_spectrum(
    spectrum: Sequence[tuple[float, float]],
    label: str,
) -> Spectrum:
    rows = sorted((float(wavelength), float(signal)) for wavelength, signal in spectrum)
    if len(rows) < 2:
        raise ValueError(f"{label} must contain at least two data points.")
    if any(
        not math.isfinite(wavelength) or not math.isfinite(signal)
        for wavelength, signal in rows
    ):
        raise ValueError(f"{label} contains a non-finite value.")
    if any(wavelength <= 0 for wavelength, _ in rows):
        raise ValueError(f"{label} wavelengths must be positive.")
    if any(
        rows[index][0] == rows[index - 1][0]
        for index in range(1, len(rows))
    ):
        raise ValueError(f"{label} contains duplicate wavelengths.")
    return rows


def _interpolate(curve: Spectrum, wavelength_nm: float, label: str) -> float:
    if wavelength_nm < curve[0][0] or wavelength_nm > curve[-1][0]:
        raise ValueError(f"{label} does not cover the common wavelength range.")
    for (lower_wavelength, lower_value), (upper_wavelength, upper_value) in zip(
        curve, curve[1:]
    ):
        if wavelength_nm == lower_wavelength:
            return lower_value
        if wavelength_nm <= upper_wavelength:
            fraction = (
                (wavelength_nm - lower_wavelength)
                / (upper_wavelength - lower_wavelength)
            )
            return lower_value + fraction * (upper_value - lower_value)
    return curve[-1][1]


def _integrate_spectrum(
    spectrum: Spectrum,
    wavelengths_nm: Sequence[float],
    detector_correction: Spectrum | None,
    label: str,
) -> float:
    corrected_values: list[float] = []
    for wavelength_nm in wavelengths_nm:
        signal = _interpolate(spectrum, wavelength_nm, label)
        if detector_correction is not None:
            signal *= _interpolate(
                detector_correction,
                wavelength_nm,
                "Detector correction curve",
            )
        corrected_values.append(signal)
    return sum(
        (lower_signal + upper_signal)
        * 0.5
        * (upper_wavelength - lower_wavelength)
        for lower_wavelength, upper_wavelength, lower_signal, upper_signal in zip(
            wavelengths_nm,
            wavelengths_nm[1:],
            corrected_values,
            corrected_values[1:],
        )
    )


def calculate_apparent_light_yield(
    sample_spectrum: Sequence[tuple[float, float]],
    reference_spectrum: Sequence[tuple[float, float]],
    reference_light_yield_photons_per_mev: float,
    detector_correction: Sequence[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Scale a sample's integrated emission by a reference's known light yield."""
    sample = _validate_spectrum(sample_spectrum, "Sample spectrum")
    reference = _validate_spectrum(reference_spectrum, "Reference spectrum")
    if (
        not math.isfinite(reference_light_yield_photons_per_mev)
        or reference_light_yield_photons_per_mev <= 0
    ):
        raise ValueError("Reference light yield must be positive and finite.")

    correction = (
        _validate_spectrum(detector_correction, "Detector correction curve")
        if detector_correction is not None
        else None
    )
    if correction is not None and any(value <= 0 for _, value in correction):
        raise ValueError("Detector correction values must be positive.")

    lower_wavelength = max(sample[0][0], reference[0][0])
    upper_wavelength = min(sample[-1][0], reference[-1][0])
    if lower_wavelength >= upper_wavelength:
        raise ValueError("Sample and reference spectra have no common wavelength range.")
    if correction is not None and (
        correction[0][0] > lower_wavelength
        or correction[-1][0] < upper_wavelength
    ):
        raise ValueError(
            "Detector correction curve must cover the common sample/reference "
            "wavelength range."
        )

    wavelengths_nm = sorted(
        {
            lower_wavelength,
            upper_wavelength,
            *(
                wavelength
                for spectrum in (sample, reference, correction or [])
                for wavelength, _ in spectrum
                if lower_wavelength < wavelength < upper_wavelength
            ),
        }
    )
    sample_signal = _integrate_spectrum(
        sample, wavelengths_nm, correction, "Sample spectrum"
    )
    reference_signal = _integrate_spectrum(
        reference, wavelengths_nm, correction, "Reference spectrum"
    )
    if not math.isfinite(sample_signal) or sample_signal <= 0:
        raise ValueError("Integrated sample signal must be positive and finite.")
    if not math.isfinite(reference_signal) or reference_signal <= 0:
        raise ValueError("Integrated reference signal must be positive and finite.")

    return {
        "sample_integrated_signal": sample_signal,
        "reference_integrated_signal": reference_signal,
        "apparent_light_yield_photons_per_mev": (
            reference_light_yield_photons_per_mev
            * sample_signal
            / reference_signal
        ),
        "wavelength_min_nm": lower_wavelength,
        "wavelength_max_nm": upper_wavelength,
    }


def calculate_absorption_corrected_light_yield(
    apparent_light_yield_photons_per_mev: float,
    sample_absorption_fraction: float,
    reference_absorption_fraction: float,
) -> float:
    """Correct a reference-relative yield for sample and reference X-ray attenuation."""
    values = (
        apparent_light_yield_photons_per_mev,
        sample_absorption_fraction,
        reference_absorption_fraction,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Light yield and absorption fractions must be finite.")
    if apparent_light_yield_photons_per_mev <= 0:
        raise ValueError("Apparent light yield must be positive.")
    for label, fraction in (
        ("Sample", sample_absorption_fraction),
        ("Reference", reference_absorption_fraction),
    ):
        if fraction <= 0 or fraction > 1:
            raise ValueError(
                f"{label} X-ray absorption fraction must be greater than 0 "
                "and at most 1."
            )
    return (
        apparent_light_yield_photons_per_mev
        * reference_absorption_fraction
        / sample_absorption_fraction
    )


def calculate_monoenergetic_absorption_fraction(
    linear_attenuation_coefficient_per_cm: float,
    path_length_mm: float,
) -> float:
    """Estimate primary-beam attenuation for a monoenergetic, normal-incidence beam."""
    if (
        not math.isfinite(linear_attenuation_coefficient_per_cm)
        or linear_attenuation_coefficient_per_cm <= 0
    ):
        raise ValueError("Linear attenuation coefficient must be positive and finite.")
    if not math.isfinite(path_length_mm) or path_length_mm <= 0:
        raise ValueError("X-ray path length must be positive and finite.")
    optical_depth = linear_attenuation_coefficient_per_cm * path_length_mm / 10
    return -math.expm1(-optical_depth)


def calculate_scintillator_equivalent_thickness(
    total_thickness_mm: float,
    scintillator_mass_fraction: float,
    scintillator_density_g_cm3: float,
    matrix_density_g_cm3: float,
) -> float:
    """Calculate the SI's scintillator-equivalent thickness for a composite layer."""
    values = (
        total_thickness_mm,
        scintillator_mass_fraction,
        scintillator_density_g_cm3,
        matrix_density_g_cm3,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("SET inputs must be finite.")
    if total_thickness_mm <= 0:
        raise ValueError("Total layer thickness must be positive.")
    if not 0 < scintillator_mass_fraction <= 1:
        raise ValueError(
            "Scintillator mass fraction must be greater than 0 and at most 1."
        )
    if scintillator_density_g_cm3 <= 0 or matrix_density_g_cm3 <= 0:
        raise ValueError("Component densities must be positive.")

    scintillator_volume = scintillator_mass_fraction / scintillator_density_g_cm3
    matrix_volume = (1 - scintillator_mass_fraction) / matrix_density_g_cm3
    return total_thickness_mm * scintillator_volume / (
        scintillator_volume + matrix_volume
    )
