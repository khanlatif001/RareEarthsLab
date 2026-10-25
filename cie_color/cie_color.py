"""CIE 1931 2 degree chromaticity calculations for emission spectra."""
from __future__ import annotations

from io import BytesIO
from typing import Mapping, Sequence

import numpy as np
import colour
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.path import Path as MatplotlibPath
from matplotlib.ticker import MultipleLocator


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


def create_cie_chromaticity_image(
    samples: Sequence[Mapping[str, str | float]],
) -> bytes:
    """Create a numbered CIE 1931 chromaticity diagram as a PNG."""
    if not samples:
        raise ValueError("At least one valid CIE coordinate is required to plot.")

    observer = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    observer_wavelengths = np.asarray(observer.domain, dtype=float)
    matching_functions = np.asarray(observer.values, dtype=float)
    visible = (observer_wavelengths >= 380.0) & (observer_wavelengths <= 780.0)
    visible_wavelengths = observer_wavelengths[visible]
    visible_functions = matching_functions[visible]
    function_sum = visible_functions.sum(axis=1)
    locus_xy = visible_functions[:, :2] / function_sum[:, np.newaxis]
    locus = MatplotlibPath(locus_xy, closed=True)

    x_values = np.linspace(0.0, 0.8, 321)
    y_values = np.linspace(0.0, 0.9, 361)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    xy_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = locus.contains_points(xy_points)
    safe_y = np.maximum(xy_points[:, 1], 1e-9)
    xyz = np.column_stack(
        (
            xy_points[:, 0] / safe_y,
            np.ones(len(xy_points)),
            np.maximum(1.0 - xy_points[:, 0] - xy_points[:, 1], 0.0) / safe_y,
        )
    )
    rgb = np.clip(colour.XYZ_to_sRGB(xyz), 0.0, 1.0)
    rgba = np.column_stack((rgb, inside.astype(float))).reshape(
        len(y_values), len(x_values), 4
    )

    figure, axis = plt.subplots(figsize=(8, 7), facecolor="black")
    canvas = FigureCanvasAgg(figure)
    axis.set_facecolor("black")
    axis.imshow(
        rgba,
        origin="lower",
        extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
        interpolation="bilinear",
        aspect="equal",
    )
    axis.plot(locus_xy[:, 0], locus_xy[:, 1], color="white", linewidth=0.8)
    axis.plot(
        [locus_xy[-1, 0], locus_xy[0, 0]],
        [locus_xy[-1, 1], locus_xy[0, 1]],
        color="white",
        linewidth=0.8,
    )

    for wavelength in (450, 470, 480, 520, 540, 560, 580, 600, 620, 700):
        index = int(np.argmin(np.abs(visible_wavelengths - wavelength)))
        x, y = locus_xy[index]
        axis.annotate(
            str(wavelength),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            color="white",
            fontsize=8,
        )

    for sample in samples:
        number = int(sample["No"])
        x = float(sample["cie_x"])
        y = float(sample["cie_y"])
        if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x + y > 1:
            raise ValueError(f"Invalid CIE xy coordinate for point {number}.")
        axis.scatter(
            [x],
            [y],
            s=115,
            color="#202020",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        axis.annotate(
            str(number),
            (x, y),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=5,
        )

    axis.set_xlim(0.0, 0.8)
    axis.set_ylim(0.0, 0.9)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_xlabel("CIE x", color="white")
    axis.set_ylabel("CIE y", color="white")
    axis.set_title("CIE 1931 Chromaticity Diagram", color="white", fontweight="bold")
    axis.tick_params(colors="white")
    for spine in axis.spines.values():
        spine.set_color("white")
    figure.tight_layout()

    output = BytesIO()
    canvas.print_png(output)
    plt.close(figure)
    return output.getvalue()
