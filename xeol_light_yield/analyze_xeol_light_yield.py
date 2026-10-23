#!/usr/bin/env python
"""Calculate absolute XEOL light yield from an absolutely calibrated spectrum."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence


PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299792458.0
Spectrum = list[tuple[float, float]]


def _split_fields(line: str) -> list[str]:
    return [field for field in re.split(r"[\s,;\t]+", line.strip()) if field]


def _validate_spectrum(spectrum: Sequence[tuple[float, float]], label: str) -> Spectrum:
    rows = sorted((float(x), float(y)) for x, y in spectrum)
    if len(rows) < 2:
        raise ValueError(f"{label} must contain at least two data points.")
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in rows):
        raise ValueError(f"{label} contains a non-finite value.")
    if any(x <= 0 for x, _ in rows):
        raise ValueError(f"{label} wavelengths must be positive.")
    if any(rows[index][0] == rows[index - 1][0] for index in range(1, len(rows))):
        raise ValueError(f"{label} contains duplicate wavelengths.")
    return rows


def read_spectrum(path: str | Path) -> Spectrum:
    """Read wavelength (nm) and signal columns, skipping leading instrument metadata."""
    rows: Spectrum = []
    data_started = False
    with Path(path).open("r", encoding="utf-8-sig", newline="") as spectrum_file:
        for line_number, line in enumerate(spectrum_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = _split_fields(line)
            try:
                wavelength_nm, signal = float(fields[0]), float(fields[1])
            except (ValueError, IndexError):
                if data_started:
                    raise ValueError(
                        f"Expected numeric wavelength and signal columns on line {line_number}."
                    )
                continue
            data_started = True
            rows.append((wavelength_nm, signal))
    return _validate_spectrum(rows, "Spectrum")


def _read_calibration_table(
    path: str | Path,
    value_column: str,
    label: str,
) -> Spectrum:
    """Read a named wavelength/value table without guessing calibration units."""
    wavelength_column = "wavelength_nm"
    rows: Spectrum = []
    column_indexes: tuple[int, int] | None = None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as table_file:
        for line_number, line in enumerate(table_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip().lower() for field in _split_fields(line)]
            normalized = [field.replace(" ", "_") for field in fields]
            if column_indexes is None:
                if wavelength_column in normalized and value_column in normalized:
                    column_indexes = (
                        normalized.index(wavelength_column),
                        normalized.index(value_column),
                    )
                continue
            try:
                wavelength_nm = float(fields[column_indexes[0]])
                value = float(fields[column_indexes[1]])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid {label} data on line {line_number}; expected "
                    f"{wavelength_column} and {value_column} columns."
                ) from exc
            rows.append((wavelength_nm, value))

    if column_indexes is None:
        raise ValueError(
            f"{label} table must have columns '{wavelength_column}' and "
            f"'{value_column}'."
        )
    return _validate_spectrum(rows, label)


def read_spectral_response(path: str | Path) -> Spectrum:
    """Read W/nm per count/s calibration factors from a CSV/TSV file."""
    curve = _read_calibration_table(
        path,
        "watts_per_nm_per_count_rate",
        "Spectral response calibration",
    )
    if any(value <= 0 for _, value in curve):
        raise ValueError("Spectral response calibration values must be positive.")
    return curve


def read_collection_efficiency(path: str | Path) -> Spectrum:
    """Read wavelength-dependent fractions of emitted light collected (0 < eta <= 1)."""
    curve = _read_calibration_table(
        path,
        "collection_efficiency",
        "Collection-efficiency calibration",
    )
    if any(value <= 0 or value > 1 for _, value in curve):
        raise ValueError("Collection efficiencies must be greater than 0 and at most 1.")
    return curve


def _interpolate(curve: Sequence[tuple[float, float]], x: float, label: str) -> float:
    if x < curve[0][0] or x > curve[-1][0]:
        raise ValueError(
            f"{label} does not cover the full spectrum range "
            f"({curve[0][0]:g}-{curve[-1][0]:g} nm). Extrapolation is not allowed."
        )
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x == x0:
            return y0
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def subtract_background(spectrum: Spectrum, background: Spectrum) -> Spectrum:
    """Subtract an interpolated background spectrum while preserving signed residuals."""
    sample = _validate_spectrum(spectrum, "Spectrum")
    background = _validate_spectrum(background, "Background spectrum")
    return [
        (wavelength, signal - _interpolate(background, wavelength, "Background spectrum"))
        for wavelength, signal in sample
    ]


def correct_spectral_response(
    spectrum: Spectrum,
    response_curve: Spectrum,
    integration_time_s: float | None,
    signal_basis: str = "counts_per_integration",
) -> Spectrum:
    """Convert detector signal to spectral power in W/nm."""
    spectrum = _validate_spectrum(spectrum, "Spectrum")
    response_curve = _validate_spectrum(response_curve, "Spectral response calibration")
    if signal_basis not in {"counts_per_integration", "counts_per_second"}:
        raise ValueError(
            "signal_basis must be 'counts_per_integration' or 'counts_per_second'."
        )
    if signal_basis == "counts_per_integration":
        if integration_time_s is None or not math.isfinite(integration_time_s) or integration_time_s <= 0:
            raise ValueError(
                "A positive integration time is required for counts_per_integration."
            )
        count_rate = lambda signal: signal / integration_time_s
    else:
        count_rate = lambda signal: signal
    return [
        (
            wavelength,
            count_rate(signal)
            * _interpolate(response_curve, wavelength, "Spectral response calibration"),
        )
        for wavelength, signal in spectrum
    ]


def _photon_rate_spectrum(spectral_power: Spectrum) -> Spectrum:
    spectral_power = _validate_spectrum(spectral_power, "Spectral power")
    return [
        (
            wavelength_nm,
            power_w_per_nm
            * wavelength_nm
            * 1e-9
            / (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S),
        )
        for wavelength_nm, power_w_per_nm in spectral_power
    ]


def _trapezoidal_integral(spectrum: Spectrum) -> float:
    return sum(
        (y0 + y1) * 0.5 * (x1 - x0)
        for (x0, y0), (x1, y1) in zip(spectrum, spectrum[1:])
    )


def calculate_detected_photon_rate(spectral_power: Spectrum) -> float:
    """Integrate the calibrated spectrum to detected photons per second."""
    return _trapezoidal_integral(_photon_rate_spectrum(spectral_power))


def correct_collection_efficiency(
    detected_photon_rate_spectrum: Spectrum,
    collection_efficiency: float | Spectrum,
) -> Spectrum:
    """Correct a photon-rate spectrum wavelength-by-wavelength for collection."""
    photon_spectrum = _validate_spectrum(
        detected_photon_rate_spectrum, "Detected photon-rate spectrum"
    )
    if isinstance(collection_efficiency, (int, float)):
        eta = float(collection_efficiency)
        if not math.isfinite(eta) or eta <= 0 or eta > 1:
            raise ValueError("Collection efficiency must be greater than 0 and at most 1.")
        return [(wavelength, rate / eta) for wavelength, rate in photon_spectrum]

    efficiency_curve = _validate_spectrum(collection_efficiency, "Collection efficiency")
    if any(eta <= 0 or eta > 1 for _, eta in efficiency_curve):
        raise ValueError("Collection efficiencies must be greater than 0 and at most 1.")
    return [
        (
            wavelength,
            rate / _interpolate(efficiency_curve, wavelength, "Collection-efficiency calibration"),
        )
        for wavelength, rate in photon_spectrum
    ]


def calculate_deposited_xray_energy(
    incident_flux_photons_s: float,
    exposure_time_s: float,
    xray_energy_kev: float,
    deposited_fraction: float,
) -> float:
    """Estimate deposited energy in keV from measured flux and a supplied fraction."""
    values = (incident_flux_photons_s, exposure_time_s, xray_energy_kev, deposited_fraction)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("X-ray inputs must be finite.")
    if incident_flux_photons_s <= 0 or exposure_time_s <= 0 or xray_energy_kev <= 0:
        raise ValueError("Flux, exposure time, and X-ray energy must be positive.")
    if deposited_fraction <= 0 or deposited_fraction > 1:
        raise ValueError("Deposited fraction must be greater than 0 and at most 1.")
    return incident_flux_photons_s * exposure_time_s * xray_energy_kev * deposited_fraction


def calculate_absolute_light_yield(
    spectral_power: Spectrum,
    exposure_time_s: float,
    collection_efficiency: float | Spectrum,
    deposited_energy_kev: float,
) -> dict[str, float]:
    """Return detected/emitted photons and absolute photons per deposited keV."""
    if not math.isfinite(exposure_time_s) or exposure_time_s <= 0:
        raise ValueError("Exposure time must be positive and finite.")
    if not math.isfinite(deposited_energy_kev) or deposited_energy_kev <= 0:
        raise ValueError("Deposited X-ray energy must be positive and finite.")
    detected_spectrum = _photon_rate_spectrum(spectral_power)
    emitted_spectrum = correct_collection_efficiency(
        detected_spectrum, collection_efficiency
    )
    detected_rate = _trapezoidal_integral(detected_spectrum)
    emitted_rate = _trapezoidal_integral(emitted_spectrum)
    if detected_rate <= 0 or emitted_rate <= 0:
        raise ValueError(
            "Background-corrected photon rate must be positive over the measured range."
        )
    emitted_photons = emitted_rate * exposure_time_s
    return {
        "detected_photon_rate_s-1": detected_rate,
        "emitted_photon_rate_s-1": emitted_rate,
        "emitted_photons": emitted_photons,
        "light_yield_photons_per_keV": emitted_photons / deposited_energy_kev,
    }


def _uncertainty_summary(
    light_yield: float,
    uncertainties_percent: dict[str, float | None],
) -> dict[str, object]:
    missing = [name for name, value in uncertainties_percent.items() if value is None]
    for name, value in uncertainties_percent.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Uncertainty for {name} must be finite and non-negative.")
    if missing:
        return {
            "combined_relative_standard_uncertainty_percent": None,
            "light_yield_standard_uncertainty_photons_per_keV": None,
            "uncertainties_not_provided": missing,
        }
    combined_percent = math.sqrt(
        sum(float(value) ** 2 for value in uncertainties_percent.values())
    )
    return {
        "combined_relative_standard_uncertainty_percent": combined_percent,
        "light_yield_standard_uncertainty_photons_per_keV": (
            light_yield * combined_percent / 100
        ),
        "uncertainties_not_provided": [],
    }


def generate_xeol_report(
    spectrum_path: str | Path,
    response_path: str | Path,
    output_path: str | Path,
    sample_metadata: dict[str, str | float],
    xray_inputs: dict[str, float],
    light_yield_results: dict[str, float],
    uncertainties_percent: dict[str, float | None],
    background_path: str | Path | None = None,
    collection_efficiency_description: str | None = None,
    acquisition_metadata: dict[str, str | float] | None = None,
    wavelength_range_nm: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Create and write a machine-readable report of the analysis inputs and result."""
    if wavelength_range_nm is None:
        wavelengths = [wavelength for wavelength, _ in read_spectrum(spectrum_path)]
        wavelength_range_nm = (min(wavelengths), max(wavelengths))
    yield_value = light_yield_results["light_yield_photons_per_keV"]
    report: dict[str, object] = {
        "method": (
            "Spectral radiometric calibration followed by wavelength-resolved "
            "photon conversion, adapted from de Haas, Dorenbos, and van Eijk (2005)."
        ),
        "spectrum_file": Path(spectrum_path).name,
        "background_file": Path(background_path).name if background_path else None,
        "measured_wavelength_range_nm": list(wavelength_range_nm),
        "sample": sample_metadata,
        "calibration": {
            "radiometric_calibration_identifier": sample_metadata[
                "radiometric_calibration_identifier"
            ],
            "wavelength_calibration_identifier": sample_metadata.get(
                "wavelength_calibration_identifier"
            ),
            "absolute_spectral_response_file": Path(response_path).name,
            "response_units": "W/nm per count/s",
            "wavelength_only_certificate_note": (
                "A wavelength calibration does not establish absolute radiometric response."
            ),
        },
        "collection_efficiency": collection_efficiency_description,
        "acquisition": acquisition_metadata or {},
        "xray": xray_inputs,
        "results": light_yield_results,
        "uncertainty": {
            **_uncertainty_summary(yield_value, uncertainties_percent),
            "input_relative_standard_uncertainties_percent": uncertainties_percent,
        },
        "assumptions_and_limits": [
            "The response table must be an absolute calibration for the complete detector chain and use the documented units.",
            "Collection efficiency is a separate fraction of all sample-emitted photons reaching the calibrated collection chain.",
            "PMT-specific quantum-efficiency and reflectivity corrections from the reference paper are not applied to the QEPro.",
            "Deposited fraction is supplied by the user and is not inferred from primary-beam attenuation.",
            "The entered exposure duration is the actual beam-on duration; scan count and integration time are not multiplied automatically.",
            "The same exposure duration is used for emitted photons and deposited X-ray energy, so its uncertainty cancels from their ratio.",
            "Yield uncertainty combines independent relative standard uncertainties for response, collection efficiency, flux, X-ray energy, and deposited fraction; counting statistics are excluded unless represented in supplied inputs.",
        ],
    }
    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate absolute XEOL light yield (photons/keV deposited)."
    )
    parser.add_argument("--spectrum", required=True, help="OceanView spectrum, wavelength_nm and signal columns.")
    parser.add_argument("--response", required=True, help="Absolute response CSV: wavelength_nm, watts_per_nm_per_count_rate.")
    parser.add_argument("--output", default="xeol_light_yield_report.json", help="JSON report path.")
    parser.add_argument("--background", help="Optional background spectrum measured with matching settings.")
    collection_group = parser.add_mutually_exclusive_group(required=True)
    collection_group.add_argument("--collection-efficiency", type=_fraction, help="Wavelength-independent collection fraction (0 < eta <= 1).")
    collection_group.add_argument("--collection-efficiency-file", help="CSV: wavelength_nm, collection_efficiency.")
    parser.add_argument("--signal-basis", choices=("counts_per_integration", "counts_per_second"), default="counts_per_integration")
    parser.add_argument("--integration-time-s", type=_positive_float, help="Required for counts_per_integration.")
    parser.add_argument("--exposure-time-s", type=_positive_float, required=True, help="Actual beam-on acquisition duration, not scans multiplied by integration time.")
    parser.add_argument("--incident-flux-photons-s", type=_positive_float, required=True)
    parser.add_argument("--xray-energy-kev", type=_positive_float, required=True)
    parser.add_argument("--deposited-fraction", type=_fraction, required=True, help="User-supplied fraction of incident X-ray energy deposited in the sample.")
    parser.add_argument("--sample-composition", required=True)
    parser.add_argument("--sample-thickness-mm", type=_positive_float, required=True)
    parser.add_argument("--collection-geometry", required=True)
    parser.add_argument(
        "--radiometric-calibration-identifier",
        required=True,
        help="Identifier for the separate absolute radiometric response calibration.",
    )
    parser.add_argument(
        "--wavelength-calibration-identifier",
        help="Optional ID of the QEPro wavelength calibration certificate.",
    )
    parser.add_argument("--calibration-uncertainty-percent", type=_nonnegative_float)
    parser.add_argument("--collection-uncertainty-percent", type=_nonnegative_float)
    parser.add_argument("--flux-uncertainty-percent", type=_nonnegative_float)
    parser.add_argument("--xray-energy-uncertainty-percent", type=_nonnegative_float)
    parser.add_argument("--deposited-fraction-uncertainty-percent", type=_nonnegative_float)
    args = parser.parse_args()

    try:
        spectrum = read_spectrum(args.spectrum)
        if args.background:
            spectrum = subtract_background(spectrum, read_spectrum(args.background))
        response_curve = read_spectral_response(args.response)
        spectral_power = correct_spectral_response(
            spectrum,
            response_curve,
            args.integration_time_s,
            args.signal_basis,
        )
        if args.collection_efficiency_file:
            collection_efficiency: float | Spectrum = read_collection_efficiency(
                args.collection_efficiency_file
            )
            collection_description = (
                f"Wavelength-dependent calibration: "
                f"{Path(args.collection_efficiency_file).name}"
            )
        else:
            collection_efficiency = args.collection_efficiency
            collection_description = f"Scalar fraction: {collection_efficiency:g}"

        deposited_energy_kev = calculate_deposited_xray_energy(
            args.incident_flux_photons_s,
            args.exposure_time_s,
            args.xray_energy_kev,
            args.deposited_fraction,
        )
        results = calculate_absolute_light_yield(
            spectral_power,
            args.exposure_time_s,
            collection_efficiency,
            deposited_energy_kev,
        )
        sample_metadata: dict[str, str | float] = {
            "composition": args.sample_composition,
            "thickness_mm": args.sample_thickness_mm,
            "collection_geometry": args.collection_geometry,
            "radiometric_calibration_identifier": args.radiometric_calibration_identifier,
            "wavelength_calibration_identifier": (
                args.wavelength_calibration_identifier or "not provided"
            ),
        }
        xray_inputs = {
            "incident_flux_photons_s": args.incident_flux_photons_s,
            "photon_energy_keV": args.xray_energy_kev,
            "actual_exposure_time_s": args.exposure_time_s,
            "deposited_fraction_user_supplied": args.deposited_fraction,
            "estimated_deposited_energy_keV": deposited_energy_kev,
        }
        uncertainties_percent = {
            "absolute spectral response": args.calibration_uncertainty_percent,
            "collection efficiency": args.collection_uncertainty_percent,
            "incident flux": args.flux_uncertainty_percent,
            "X-ray energy": args.xray_energy_uncertainty_percent,
            "deposited fraction": args.deposited_fraction_uncertainty_percent,
        }
        generate_xeol_report(
            args.spectrum,
            args.response,
            args.output,
            sample_metadata,
            xray_inputs,
            results,
            uncertainties_percent,
            args.background,
            collection_description,
            {
                "signal_basis": args.signal_basis,
                "integration_time_s": args.integration_time_s or "not applicable",
            },
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Absolute XEOL light yield: {results['light_yield_photons_per_keV']:.6g} photons/keV")
    print(f"Estimated deposited energy: {deposited_energy_kev:.6g} keV")
    uncertainty_summary = _uncertainty_summary(
        results["light_yield_photons_per_keV"],
        uncertainties_percent,
    )
    uncertainty_value = uncertainty_summary[
        "light_yield_standard_uncertainty_photons_per_keV"
    ]
    if uncertainty_value is None:
        print("Combined uncertainty not calculated; provide all five uncertainty inputs.")
    else:
        print(f"Combined standard uncertainty: ±{uncertainty_value:.6g} photons/keV")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
