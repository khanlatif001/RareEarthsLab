#!/usr/bin/env python
"""Detect spectral peaks and compare photon energies with Carnall energy levels.

IMPORTANT: an emission peak is a transition energy E_upper - E_lower, not
normally an absolute level energy. The script therefore compares each peak's
wavenumber with pairwise differences of levels in the database.

The included Carnall JSON is a partial starter transcription. Only explicitly
entered levels are searchable; empty/incomplete entries cannot support reliable
identification. Review all assignments against the scanned source tables.
"""
from __future__ import annotations
import argparse, json
import hashlib
import re
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
import matplotlib.pyplot as plt
import streamlit as st
from cie_color import (
    calculate_cie_coordinates,
    create_cie_chromaticity_image,
)
from Judd_Ofelt.omega_parameters import (
    Inputs as JuddOfeltInputs,
    calculate as calculate_judd_ofelt,
)
from xeol_light_yield.apparent_light_yield import (
    calculate_absorption_corrected_light_yield,
    calculate_apparent_light_yield,
    calculate_monoenergetic_absorption_fraction,
    calculate_scintillator_equivalent_thickness,
)
from xeol_light_yield.analyze_xeol_light_yield import (
    calculate_absolute_light_yield,
    calculate_deposited_xray_energy,
    correct_spectral_response,
    generate_xeol_report,
    read_collection_efficiency,
    read_spectral_response,
    read_spectrum as read_xeol_spectrum,
    subtract_background,
)


BACKGROUND_COLORMAP_OPTIONS = {
    "Emission spectra — Inferno": "inferno",
    "Emission spectra — Magma": "magma",
    "Emission spectra — Plasma": "plasma",
    "Emission spectra — Viridis": "viridis",
    "Emission spectra — Turbo": "turbo",
    "Emission spectra — Hot": "hot",
    "Emission spectra — Cividis": "cividis",
    "Scientific comparison — Coolwarm": "coolwarm",
    "Scientific comparison — Seismic": "seismic",
    "Scientific comparison — RdYlBu": "RdYlBu",
    "Scientific comparison — Spectral": "Spectral",
    "Intensity maps — YlGnBu": "YlGnBu",
    "Intensity maps — YlOrRd": "YlOrRd",
    "Intensity maps — Purples": "Purples",
    "Intensity maps — Ocean": "ocean",
    "Intensity maps — Terrain": "terrain",
    "Intensity maps — Gist Earth": "gist_earth",
    "Specialized — Rainbow": "rainbow",
    "Specialized — HSV": "hsv",
    "Specialized — Cool": "cool",
    "Specialized — Twilight": "twilight",
    "Specialized — Twilight Shifted": "twilight_shifted",
}


def read_spectrum(path, axis="auto"):
    """Read numeric spectrum rows from whitespace/comma/semicolon text files."""
    spectrum_rows = []
    data_started = False
    with Path(path).open("r", encoding="utf-8-sig") as spectrum_file:
        for line_number, line in enumerate(spectrum_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field for field in re.split(r"[\s,;]+", line) if field]
            if len(fields) < 2:
                if data_started:
                    raise ValueError(
                        f"Expected two numeric spectrum columns on line {line_number}."
                    )
                continue
            try:
                x_value, intensity = float(fields[0]), float(fields[1])
            except ValueError as exc:
                if data_started:
                    raise ValueError(
                        f"Expected two numeric spectrum columns on line {line_number}."
                    ) from exc
                continue
            data_started = True
            spectrum_rows.append((x_value, intensity))

    df = pd.DataFrame(spectrum_rows, columns=["x", "intensity"])
    df = df.dropna().sort_values("x").drop_duplicates("x").reset_index(drop=True)
    if len(df) < 5:
        raise ValueError("Spectrum contains fewer than five valid data points.")
    x = df["x"].to_numpy(float)
    if axis == "auto":
        median_x = float(np.median(x))
        axis = "energy_ev" if median_x < 20 else (
            "wavenumber_cm1" if median_x > 2500 else "wavelength_nm")
    if axis == "wavelength_nm":
        if np.any(x <= 0): raise ValueError("Wavelengths must be positive.")
        df["wavelength_nm"] = x
        df["wavenumber_cm1"] = 1e7 / x
        label = "Wavelength (nm)"
    elif axis == "wavenumber_cm1":
        if np.any(x <= 0): raise ValueError("Wavenumbers must be positive.")
        df["wavenumber_cm1"] = x
        df["wavelength_nm"] = 1e7 / x
        label = "Wavenumber (cm$^{-1}$)"
    elif axis == "energy_ev":
        if np.any(x <= 0): raise ValueError("Photon energies must be positive.")
        df["wavenumber_cm1"] = x * 8065.544005
        df["wavelength_nm"] = 1239.841984 / x
        label = "Photon energy (eV)"
    else:
        raise ValueError(f"Unsupported axis {axis!r}")
    df.attrs["axis_label"] = label
    df.attrs["axis"] = axis
    return df


def choose_spectrum():
    """Open a native file picker when no spectrum path is supplied."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Pass a spectrum path; the file picker is unavailable.") from exc
    root = tk.Tk()
    root.withdraw()
    try:
        path = filedialog.askopenfilename(
            title="Choose an emission spectrum",
            filetypes=[("Spectrum data", "*.txt *.csv *.dat"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    if not path:
        raise SystemExit("No spectrum selected.")
    return Path(path)


def infer_ion_from_filename(path, database):
    """Use an ion symbol present in the filename when possible."""
    stem = Path(path).stem.lower()
    for ion in database:
        symbol = re.sub(r"3\+$", "", ion).lower()
        if symbol in stem:
            return ion
    return None


def format_state(state):
    """Format common term symbols for Matplotlib, keeping unknown labels intact."""
    match = re.fullmatch(r"(\d+)([A-Za-z])(\d+(?:/\d+)?)", str(state))
    if not match:
        return str(state)
    multiplicity, term, angular_momentum = match.groups()
    return rf"$^{{{multiplicity}}}{term}_{{{angular_momentum}}}$"


def format_ion(ion):
    """Format an ion charge as a superscript for Matplotlib."""
    match = re.fullmatch(r"([A-Za-z]+)(\d+[+-])", str(ion))
    if not match:
        return str(ion)
    element, charge = match.groups()
    return rf"{element}$^{{{charge}}}$"


def detect_peaks(df, axis="wavelength_nm", smooth_window= 9, polyorder=3,
                 prominence=0.02, distance_points=5, min_height=None):
    """Detect local maxima after optional Savitzky-Golay smoothing.

    prominence is an absolute intensity value unless --relative-prominence is
    selected in the CLI, in which case it is a fraction of the signal range.
    """
    y = df["intensity"].to_numpy(float)
    # Ensure valid odd window smaller than data length.
    win = max(5, int(smooth_window))
    if win % 2 == 0: win += 1
    if win >= len(y): win = len(y) - 1 if (len(y)-1) % 2 else len(y)-2
    win = max(5, win)
    ys = savgol_filter(y, window_length=win, polyorder=min(polyorder, win-2))
    peaks, props = find_peaks(ys, prominence=prominence, distance=max(1, int(distance_points)),
                              height=min_height)
    width_results = peak_widths(ys, peaks, rel_height=0.5) if len(peaks) else None
    widths = width_results[0] if width_results is not None else np.array([])
    # Robust estimate of high-frequency residual noise; uncertainty is approximate,
    # intended as a diagnostic rather than a formal fit-derived confidence interval.
    residual = y - ys
    noise_sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    x_axis = df["x"].to_numpy(float)
    dx = float(np.median(np.diff(x_axis)))
    left_x = (np.interp(width_results[2], np.arange(len(x_axis)), x_axis)
              if width_results is not None else np.array([]))
    right_x = (np.interp(width_results[3], np.arange(len(x_axis)), x_axis)
               if width_results is not None else np.array([]))
    out = []
    for i, p in enumerate(peaks):
        row = {
            "peak_id": int(i+1),
            "x_peak": float(df.iloc[p]["x"]),
            "wavelength_nm": float(df.iloc[p]["wavelength_nm"]),
            "wavenumber_cm1": float(df.iloc[p]["wavenumber_cm1"]),
            "intensity_raw": float(y[p]),
            "intensity_smoothed": float(ys[p]),
            "prominence": float(props["prominences"][i]),
            "width_points_fwhm_approx": float(widths[i]) if len(widths) else np.nan,
            "width_x_fwhm_approx": float(widths[i] * dx) if len(widths) else np.nan,
            "integration_x1": float(left_x[i]),
            "integration_x2": float(right_x[i]),
            "integrated_area": np.nan,
            "x_uncertainty_approx": float(max(dx / 2.0, (widths[i] * dx) * noise_sigma /
                                               max(float(props["prominences"][i]), 1e-12)))
                                    if len(widths) else dx / 2.0,
            "noise_sigma_robust": float(noise_sigma),
            "peak_index": int(p),
        }
        out.append(row)
    return ys, pd.DataFrame(out)


def integrate_spectrum_interval(df, x1, x2):
    """Integrate raw intensity between x limits, interpolating at both endpoints."""
    x = df["x"].to_numpy(float)
    intensity = df["intensity"].to_numpy(float)
    if not np.isfinite(x1) or not np.isfinite(x2) or x1 >= x2:
        raise ValueError("Integration limits must be finite and x1 must be less than x2.")
    if x1 < x[0] or x2 > x[-1]:
        raise ValueError(f"Integration limits must be within {x[0]:g} and {x[-1]:g}.")
    interior = (x > x1) & (x < x2)
    integration_x = np.concatenate(([x1], x[interior], [x2]))
    integration_y = np.interp(integration_x, x, intensity)
    return float(np.trapezoid(integration_y, x=integration_x))


def build_transition_catalog(database, energy_field="observed_cm1", include_calculated=True):
    """Build transitions from explicit records or pairwise level differences."""
    transitions = []
    for ion, info in database.items():
        explicit_transitions = info.get("transitions", [])
        for reference in explicit_transitions:
            try:
                upper_energies = reference.get("upper_energies_cm1")
                if upper_energies is None:
                    upper_energies = [reference["upper_energy_cm1"]]
                upper_energies = [float(value) for value in upper_energies]
                lower_energies = [float(value) for value in reference["lower_energies_cm1"]]
                upper_state = reference["upper_state"]
                lower_state = reference["lower_state"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid explicit transition record for {ion}: {reference!r}") from exc
            if (not all(np.isfinite(upper_energies)) or
                    not all(np.isfinite(lower_energies))):
                raise ValueError(f"Non-finite energy in explicit transition record for {ion}: {reference!r}")
            for upper_energy in dict.fromkeys(upper_energies):
                for lower_energy in dict.fromkeys(lower_energies):
                    transition_energy = upper_energy - lower_energy
                    if transition_energy <= 0:
                        raise ValueError(f"Lower level is not below upper level for {ion}: {reference!r}")
                    transitions.append({
                        "ion": ion,
                        "upper_state": upper_state,
                        "lower_state": lower_state,
                        "upper_energy_cm1": upper_energy,
                        "lower_energy_cm1": lower_energy,
                        "transition_cm1": transition_energy,
                        "energy_column": reference.get("energy_column", "explicit_transition"),
                        "upper_source_page": reference.get(
                            "upper_source_page", reference.get("source_page")
                        ),
                        "lower_source_page": reference.get(
                            "lower_source_page", reference.get("source_page")
                        ),
                        "transcription_status": reference.get("transcription_status", ""),
                    })
        if explicit_transitions:
            continue
        levels = info.get("levels", [])
        # Include observed and calculated tables separately; never mix the columns.
        fields = [energy_field]
        if include_calculated and energy_field == "observed_cm1":
            fields.append("calculated_cm1")
        for field in fields:
            available = []
            for lev in levels:
                e = lev.get(field)
                if e is None: continue
                try: e = float(e)
                except (TypeError, ValueError): continue
                if not np.isfinite(e): continue
                available.append((lev.get("state", "state?"), e, lev.get("source_page"),
                                  lev.get("transcription_status", "")))
            # Use unique rows only, preserving same-state Stark components if their energies differ.
            for (s1, e1, p1, status1), (s2, e2, p2, status2) in __import__("itertools").combinations(available, 2):
                if abs(e1-e2) < 1e-9: continue
                upper_state, lower_state = (s1, s2) if e1 > e2 else (s2, s1)
                upper_e, lower_e = max(e1, e2), min(e1, e2)
                transitions.append({
                    "ion": ion, "upper_state": upper_state, "lower_state": lower_state,
                    "upper_energy_cm1": upper_e, "lower_energy_cm1": lower_e,
                    "transition_cm1": upper_e-lower_e, "energy_column": field,
                    "upper_source_page": p1 if e1 >= e2 else p2,
                    "lower_source_page": p2 if e1 >= e2 else p1,
                    "transcription_status": "; ".join(x for x in [status1, status2] if x)
                })
    return pd.DataFrame(transitions)


def match_peaks(peaks, database, tolerance_cm1=None, max_matches=5,
                energy_field="observed_cm1", ion_filter=None, tolerance_nm=3.0):
    """Match detected peaks against transitions within a wavelength tolerance."""
    catalog = build_transition_catalog(database, energy_field=energy_field, include_calculated=True)
    if ion_filter:
        wanted = {x.strip() for x in ion_filter.split(",")}
        if "ion" in catalog:
            catalog = catalog[catalog["ion"].isin(wanted)]
        else:
            catalog = catalog.iloc[0:0]
    rows = []
    for _, peak in peaks.iterrows():
        if catalog.empty:
            row = {
                "peak_id": int(peak["peak_id"]),
                "match_rank": None,
                "status": "NO_REFERENCE_LEVELS",
                "message": "No usable structured level pairs for the selected ion(s); verify or complete the database transcription.",
            }
            if ion_filter:
                row["ion"] = ion_filter
            rows.append(row)
            continue
        catalog = catalog.copy()
        catalog["delta_cm1"] = catalog["transition_cm1"] - float(peak["wavenumber_cm1"])
        catalog["abs_delta_cm1"] = catalog["delta_cm1"].abs()
        catalog["reference_wavelength_nm"] = 1e7 / catalog["transition_cm1"]
        catalog["delta_nm"] = (catalog["reference_wavelength_nm"] -
                               float(peak["wavelength_nm"]))
        catalog["abs_delta_nm"] = catalog["delta_nm"].abs()
        if tolerance_cm1 is None:
            hits = catalog[catalog["abs_delta_nm"] <= float(tolerance_nm)].sort_values("abs_delta_nm")
            nearest = catalog.nsmallest(max_matches, "abs_delta_nm")
        else:
            hits = catalog[catalog["abs_delta_cm1"] <= float(tolerance_cm1)].sort_values("abs_delta_cm1")
            nearest = catalog.nsmallest(max_matches, "abs_delta_cm1")
        if hits.empty:
            # Save nearest candidate(s) to show how far away the closest reference is.
            hits = nearest
            status = "OUTSIDE_TOLERANCE"
        else:
            hits = hits.head(max_matches)
            status = "CANDIDATE_MATCH"
        for rank, (_, tr) in enumerate(hits.iterrows(), start=1):
            rows.append({
                "peak_id": int(peak["peak_id"]), "match_rank": rank, "status": status,
                "ion": tr["ion"], "upper_state": tr["upper_state"], "lower_state": tr["lower_state"],
                "observed_wavelength_nm": float(peak["wavelength_nm"]),
                "observed_wavenumber_cm1": float(peak["wavenumber_cm1"]),
                "reference_transition_cm1": float(tr["transition_cm1"]),
                "reference_wavelength_nm": float(tr["reference_wavelength_nm"]),
                "delta_reference_minus_observed_nm": float(tr["delta_nm"]),
                "abs_difference_nm": float(tr["abs_delta_nm"]),
                "delta_reference_minus_observed_cm1": float(tr["delta_cm1"]),
                "abs_difference_cm1": float(tr["abs_delta_cm1"]),
                "energy_column": tr["energy_column"],
                "upper_source_page": tr["upper_source_page"],
                "lower_source_page": tr["lower_source_page"],
                "transcription_status": tr["transcription_status"],
            })
    return pd.DataFrame(rows), catalog


def plot_results(df, smoothed, peaks, matches, output_png, axis_label,
                 peak_label_fontsize=10, axis_label_fontsize=16,
                 background_colormap="inferno", spectrum_linewidth=1.8,
                 smoothed_signal_color="#ff5a1f", tick_label_fontsize=12,
                 show_peak_labels=True, x_axis_limits=None, show_title=True,
                 show_provisional_note=True, y_axis_limits=None,
                 dark_background=True, show_intensity_background=False):
    x = df["x"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    background_color = "#050505" if dark_background else "white"
    foreground_color = "white" if dark_background else "black"
    grid_color = "white" if dark_background else "#777777"
    experimental_color = "#ffd447" if dark_background else "#0072B2"
    line_color = smoothed_signal_color
    fig.patch.set_facecolor(background_color)
    intensity_range = float(np.ptp(y))
    display_max = float(np.max(y))
    plot_y_limits = y_axis_limits
    if y_axis_limits is not None and y_axis_limits[0] == y_axis_limits[1]:
        padding = max(abs(y_axis_limits[0]) * 0.01, 0.5)
        plot_y_limits = (
            y_axis_limits[0] - padding,
            y_axis_limits[1] + padding,
        )
    if show_intensity_background and intensity_range > 0:
        intensity_map = np.tile((y - np.min(y)) / intensity_range, (32, 1))
        x_edges = np.empty(x.size + 1)
        x_edges[1:-1] = (x[:-1] + x[1:]) / 2
        x_edges[0] = x[0] - (x[1] - x[0]) / 2
        x_edges[-1] = x[-1] + (x[-1] - x[-2]) / 2
        background_min, background_max = (
            plot_y_limits if plot_y_limits is not None else (0, display_max * 1.08)
        )
        y_edges = np.linspace(background_min, background_max, intensity_map.shape[0] + 1)
        ax.pcolormesh(x_edges, y_edges, intensity_map, shading="flat",
                      cmap=background_colormap, alpha=1.0, zorder=0)
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=background_colormap), ax=ax, pad=0.015
        )
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Min", "Max"])
        colorbar.ax.tick_params(colors=foreground_color)
    ax.set_facecolor(background_color)
    ax.plot(x, y, lw=spectrum_linewidth, alpha=0.95, color=experimental_color,
            label="Experimental spectrum")
    ax.plot(x, smoothed, lw=spectrum_linewidth, color=line_color,
            label="Smoothed signal")
    if show_peak_labels and not peaks.empty:
        ax.scatter(peaks["x_peak"], peaks["intensity_smoothed"], s=24, zorder=4,
                   marker="o", color="#F0E442" if dark_background else "#CC79A7",
                   edgecolors=foreground_color, label="Detected peaks")
        # Label each detected peak; top candidate shown only if a plausible match exists.
        for _, p in peaks.iterrows():
            subset = matches[(matches["peak_id"] == p["peak_id"]) &
                             matches.get("status", pd.Series(dtype=str)).isin(
                                 ["CANDIDATE_MATCH"])]
            if not subset.empty and pd.notna(subset.iloc[0].get("ion")):
                m = subset.iloc[0]
                label = (f'{format_ion(m["ion"])} {format_state(m["upper_state"])} → '
                         f'{format_state(m["lower_state"])}\n'
                         f'Ref {m["reference_wavelength_nm"]:.1f} nm '
                         f'(Δ {m["delta_reference_minus_observed_nm"]:+.1f} nm)')
            else:
                label = f'Peak {int(p["peak_id"])}: no database match'
            label += f'\n{p["wavelength_nm"]:.1f} nm'
            label_above = p["intensity_smoothed"] <= display_max * 0.55
            offset = (0, 8) if label_above else (0, -8)
            vertical_alignment = "bottom" if label_above else "top"
            ax.annotate(label, (p["x_peak"], p["intensity_smoothed"]),
                        xytext=offset, textcoords="offset points",
                        fontsize=peak_label_fontsize, color=foreground_color, rotation=90,
                        bbox={"facecolor": "none", "edgecolor": "none", "pad": 1.5},
                        ha="center", va=vertical_alignment)
    if x_axis_limits is not None:
        ax.set_xlim(*x_axis_limits)
    elif x.min() < 350 and x.max() > 900 and "Wavelength" in axis_label:
        ax.set_xlim(350, 900)
    if plot_y_limits is not None:
        ax.set_ylim(*plot_y_limits)
    ax.set_xlabel(axis_label, fontsize=axis_label_fontsize)
    ax.set_ylabel("Intensity (a.u.)", fontsize=axis_label_fontsize)
    if show_title:
        title = ("Emission spectrum with peak and electronic transition labels"
                 if show_peak_labels else "Emission spectrum with detected peaks")
        ax.set_title(title, color=foreground_color)
    ax.tick_params(colors=foreground_color, labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_color(foreground_color)
    ax.xaxis.label.set_color(foreground_color)
    ax.yaxis.label.set_color(foreground_color)
    ax.grid(color=grid_color, alpha=0.2)
    if show_peak_labels:
        ax.legend(facecolor="none", edgecolor="none", framealpha=0,
                  labelcolor=foreground_color)
    transcription_status = matches.get("transcription_status", pd.Series(dtype=str)).fillna("").astype(str)
    if (show_provisional_note and
            transcription_status.str.contains("provisional", case=False).any()):
        fig.text(0.5, 0.015, "Provisional database transcription; verify before scientific use.",
                 ha="center", fontsize=8, color=foreground_color)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_spectrum_comparison(
    spectra,
    mode,
    output_png,
    axis_label,
    linewidth=1.8,
    show_scan_labels=True,
    x_axis_limits=None,
    y_axis_limits=None,
    x_label_fontsize=16,
    y_label_fontsize=16,
    x_tick_fontsize=12,
    y_tick_fontsize=12,
    legend_fontsize=10,
    dark_background=True,
    show_intensity_background=False,
    background_colormap="inferno",
):
    """Plot independent scans as an overlay or in vertically stacked axes."""
    if not spectra:
        raise ValueError("Upload at least one spectrum to plot.")
    if mode not in {"Overlay", "Stacked"}:
        raise ValueError(f"Unsupported comparison layout {mode!r}.")

    color_palette = plt.get_cmap("tab20" if dark_background else "tab10")
    figure_width_inches = 7.0
    if mode == "Overlay":
        fig, ax = plt.subplots(figsize=(figure_width_inches, 4.5))
        axes = [ax]
    else:
        fig, axes_array = plt.subplots(
            len(spectra),
            1,
            figsize=(figure_width_inches, max(4.5, 2.25 * len(spectra))),
            sharex=True,
            squeeze=False,
        )
        axes = list(axes_array[:, 0])

    background_color = "#050505" if dark_background else "white"
    foreground_color = "white" if dark_background else "black"
    fig.patch.set_facecolor(background_color)
    all_intensities = np.concatenate([
        spectrum["intensity"].to_numpy(float) for _, spectrum in spectra
    ])
    intensity_min = float(np.min(all_intensities))
    intensity_range = float(np.ptp(all_intensities))
    intensity_max = float(np.max(all_intensities))
    if intensity_range == 0:
        intensity_range = 1.0
    plot_y_min, plot_y_max = (
        y_axis_limits
        if y_axis_limits is not None
        else (intensity_min, intensity_max)
    )
    if plot_y_min == plot_y_max:
        padding = max(abs(plot_y_min) * 0.01, 0.5)
        plot_y_min -= padding
        plot_y_max += padding
    for index, (name, spectrum) in enumerate(spectra):
        ax = axes[0] if mode == "Overlay" else axes[index]
        ax.set_facecolor(background_color)
        x_values = spectrum["x"].to_numpy(float)
        y_values = spectrum["intensity"].to_numpy(float)
        if show_intensity_background:
            x_edges = np.empty(x_values.size + 1)
            x_edges[1:-1] = (x_values[:-1] + x_values[1:]) / 2
            x_edges[0] = x_values[0] - (x_values[1] - x_values[0]) / 2
            x_edges[-1] = x_values[-1] + (x_values[-1] - x_values[-2]) / 2
            intensity_map = np.tile(
                (y_values - intensity_min) / intensity_range, (32, 1)
            )
            y_edges = np.linspace(plot_y_min, plot_y_max, 33)
            ax.pcolormesh(
                x_edges,
                y_edges,
                intensity_map,
                shading="flat",
                cmap=background_colormap,
                alpha=0.3,
                zorder=0,
            )
        ax.plot(
            x_values,
            y_values,
            color=color_palette(index % (20 if dark_background else 10)),
            linewidth=linewidth,
            label=name,
        )
        ax.tick_params(
            axis="x", colors=foreground_color, labelsize=x_tick_fontsize
        )
        ax.tick_params(
            axis="y", colors=foreground_color, labelsize=y_tick_fontsize
        )
        ax.grid(color=foreground_color, alpha=0.18)
        for spine in ax.spines.values():
            spine.set_color(foreground_color)
        ax.yaxis.label.set_color(foreground_color)
        if mode == "Stacked":
            ax.set_ylabel(
                name if show_scan_labels else "Intensity (a.u.)",
                color=foreground_color,
                fontsize=legend_fontsize if show_scan_labels else y_label_fontsize,
            )
        if x_axis_limits is not None:
            ax.set_xlim(*x_axis_limits)
        if y_axis_limits is not None:
            ax.set_ylim(*y_axis_limits)

    if show_intensity_background:
        colorbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=intensity_min, vmax=intensity_min + intensity_range),
                cmap=background_colormap,
            ),
            cax=colorbar_ax,
        )
        colorbar.ax.tick_params(colors=foreground_color)

    if mode == "Overlay":
        axes[0].set_ylabel(
            "Intensity (a.u.)",
            color=foreground_color,
            fontsize=y_label_fontsize,
        )
        if show_scan_labels:
            axes[0].legend(
                facecolor="none",
                edgecolor="none",
                framealpha=0,
                labelcolor=foreground_color,
                loc="best",
                fontsize=legend_fontsize,
            )
    else:
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(
        axis_label, color=foreground_color, fontsize=x_label_fontsize
    )

    if show_intensity_background:
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.96, hspace=0.12)
    else:
        fig.tight_layout()
    fig.savefig(
        output_png,
        dpi=300,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(fig)


def render_apparent_xeol_yield(
    uploaded_file,
    spectrum_df,
    spectrum_axis,
    spectrum_identity,
):
    """Render reference-based apparent and X-ray attenuation-corrected yield."""
    with st.expander("Apparent XEOL light yield (reference method)"):
        st.markdown(
            r"""
            \[
            LY_a(s)=LY_{\mathrm{ref}}
            \frac{\int I_s(\lambda)D(\lambda)\,d\lambda}
                 {\int I_{\mathrm{ref}}(\lambda)D(\lambda)\,d\lambda}
            \]
            Upload the sample spectrum currently open in the app and a reference
            measured with comparable X-ray fluence, collection geometry, and
            detector settings. The output is on the reference's photons/MeV scale;
            it is an absolute-scale value only when the entered reference yield
            is traceable.
            """
        )
        reference_upload = st.file_uploader(
            "Reference phosphor spectrum",
            type=["txt", "csv", "dat"],
            key=f"apparent_reference_{spectrum_identity}",
            help="Use the same spectral axis and comparable measurement conditions as the sample.",
        )
        sample_background_upload = st.file_uploader(
            "Optional sample background spectrum",
            type=["txt", "csv", "dat"],
            key=f"apparent_sample_background_{spectrum_identity}",
            help="Measure with matching settings; the background must cover the sample wavelength range.",
        )
        reference_background_upload = st.file_uploader(
            "Optional reference background spectrum",
            type=["txt", "csv", "dat"],
            key=f"apparent_reference_background_{spectrum_identity}",
            help="Measure with matching settings; the background must cover the reference wavelength range.",
        )
        reference_light_yield = st.number_input(
            "Reference light yield (photons/MeV)",
            min_value=0.0,
            value=0.0,
            format="%.6g",
            key=f"apparent_reference_yield_{spectrum_identity}",
        )
        reference_status = st.selectbox(
            "Reference-value status",
            ["Certified/traceable absolute value", "Approximate or relative value"],
            key=f"apparent_reference_status_{spectrum_identity}",
        )
        reference_source = st.text_input(
            "Reference yield source (certificate or citation)",
            key=f"apparent_reference_source_{spectrum_identity}",
        )

        detector_mode = st.radio(
            "Detector-response handling",
            [
                "Upload D(λ) correction multiplier",
                "Both spectra already corrected (D = 1)",
                "No correction available (uncorrected ratio)",
            ],
            key=f"apparent_detector_mode_{spectrum_identity}",
            help=(
                "Upload a wavelength_nm, correction_multiplier table only if "
                "the spectra have not already been corrected with that function."
            ),
        )
        detector_upload = None
        if detector_mode == "Upload D(λ) correction multiplier":
            detector_upload = st.file_uploader(
                "Detector correction curve D(λ) CSV",
                type=["txt", "csv", "dat"],
                key=f"apparent_detector_curve_{spectrum_identity}",
                help="Two columns: wavelength in nm and a positive multiplicative correction factor.",
            )
        elif detector_mode == "No correction available (uncorrected ratio)":
            st.warning(
                "This ratio is not corrected for wavelength-dependent detector "
                "response and may be biased when the spectra have different shapes."
            )

        apply_absorption_correction = st.checkbox(
            "Also calculate X-ray-attenuation-corrected yield",
            key=f"apparent_apply_absorption_{spectrum_identity}",
        )
        absorption_method = None
        absorption_inputs: dict[str, float] = {}
        if apply_absorption_correction:
            st.markdown(
                r"""
                \[
                LY_{\mathrm{XAE-corr}}(s)=LY_a(s)
                \frac{F_{\mathrm{abs,ref}}}{F_{\mathrm{abs},s}}
                \]
                """
            )
            absorption_method = st.radio(
                "X-ray attenuation input",
                [
                    "Enter sample/reference attenuation fractions",
                    "Estimate from monochromatic μ and path length",
                ],
                key=f"apparent_absorption_method_{spectrum_identity}",
                help=(
                    "For a polychromatic beam, enter energy-weighted attenuation "
                    "fractions calculated from its spectrum and material coefficients."
                ),
            )
            sample_column, reference_column = st.columns(2)
            if absorption_method == "Enter sample/reference attenuation fractions":
                absorption_inputs["sample_fraction"] = sample_column.number_input(
                    "Sample X-ray attenuation fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    format="%.6f",
                    key=f"apparent_sample_fraction_{spectrum_identity}",
                    help="Fraction of incident primary X-rays attenuated in the sample.",
                )
                absorption_inputs["reference_fraction"] = reference_column.number_input(
                    "Reference X-ray attenuation fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    format="%.6f",
                    key=f"apparent_reference_fraction_{spectrum_identity}",
                    help="Fraction of incident primary X-rays attenuated in the reference.",
                )
            else:
                absorption_inputs["sample_mu"] = sample_column.number_input(
                    "Sample linear attenuation coefficient (cm⁻¹)",
                    min_value=0.0,
                    value=0.0,
                    key=f"apparent_sample_mu_{spectrum_identity}",
                )
                absorption_inputs["sample_path_mm"] = sample_column.number_input(
                    "Sample X-ray path length (mm)",
                    min_value=0.0,
                    value=0.0,
                    key=f"apparent_sample_path_{spectrum_identity}",
                )
                absorption_inputs["reference_mu"] = reference_column.number_input(
                    "Reference linear attenuation coefficient (cm⁻¹)",
                    min_value=0.0,
                    value=0.0,
                    key=f"apparent_reference_mu_{spectrum_identity}",
                )
                absorption_inputs["reference_path_mm"] = reference_column.number_input(
                    "Reference X-ray path length (mm)",
                    min_value=0.0,
                    value=0.0,
                    key=f"apparent_reference_path_{spectrum_identity}",
                )
                st.caption(
                    "This monochromatic, normal-incidence Beer–Lambert estimate "
                    "is primary-beam attenuation, not deposited-energy fraction."
                )

        calculate_set = st.checkbox(
            "Calculate the SI scintillator-equivalent thickness (SET) for a composite layer",
            key=f"apparent_calculate_set_{spectrum_identity}",
        )
        set_inputs: dict[str, float] = {}
        if calculate_set:
            set_columns = st.columns(4)
            set_inputs["total_thickness_mm"] = set_columns[0].number_input(
                "Total layer thickness h (mm)",
                min_value=0.0,
                value=0.0,
                key=f"apparent_set_h_{spectrum_identity}",
            )
            set_inputs["mass_fraction"] = set_columns[1].number_input(
                "Scintillator mass fraction wₛ",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                key=f"apparent_set_ws_{spectrum_identity}",
            )
            set_inputs["scintillator_density_g_cm3"] = set_columns[2].number_input(
                "Scintillator density ρₛ (g/cm³)",
                min_value=0.0,
                value=0.0,
                key=f"apparent_set_rhos_{spectrum_identity}",
            )
            set_inputs["matrix_density_g_cm3"] = set_columns[3].number_input(
                "Matrix density ρₚ (g/cm³)",
                min_value=0.0,
                value=0.0,
                key=f"apparent_set_rhop_{spectrum_identity}",
            )
            st.caption(
                "The SET equation follows the Supporting Information and returns "
                "d in the same length unit as h."
            )

        calculate_clicked = st.button(
            "Calculate apparent light yield",
            key=f"apparent_calculate_{spectrum_identity}",
            type="primary",
        )
        calculation_inputs = {
            "reference_light_yield_photons_per_mev": reference_light_yield,
            "reference_status": reference_status,
            "reference_source": reference_source,
            "detector_mode": detector_mode,
            "apply_absorption_correction": apply_absorption_correction,
            "absorption_method": absorption_method,
            "absorption_inputs": absorption_inputs,
            "calculate_set": calculate_set,
            "set_inputs": set_inputs,
            "spectrum_axis": spectrum_axis,
        }
        request_hash = hashlib.sha256(
            json.dumps(calculation_inputs, sort_keys=True).encode("utf-8")
        )
        for uploaded_item in (
            uploaded_file,
            reference_upload,
            sample_background_upload,
            reference_background_upload,
            detector_upload,
        ):
            if uploaded_item is not None:
                request_hash.update(uploaded_item.getvalue())
        request_fingerprint = request_hash.hexdigest()
        result_state_key = f"apparent_xeol_result_{spectrum_identity}"
        cached_result = st.session_state.get(result_state_key)

        if calculate_clicked and not reference_upload:
            st.error("Upload a reference phosphor spectrum.")
            return
        if calculate_clicked and reference_light_yield <= 0:
            st.error("Enter a positive reference light yield.")
            return
        if (
            calculate_clicked
            and reference_status == "Certified/traceable absolute value"
            and not reference_source.strip()
        ):
            st.error("Provide the certificate or citation for the reference yield.")
            return
        if calculate_clicked and detector_mode == "Upload D(λ) correction multiplier" and not detector_upload:
            st.error("Upload the detector correction curve or select another correction mode.")
            return
        if calculate_clicked and apply_absorption_correction:
            if absorption_method == "Enter sample/reference attenuation fractions":
                if any(
                    value <= 0 or value > 1
                    for value in absorption_inputs.values()
                ):
                    st.error("Both attenuation fractions must be greater than 0 and at most 1.")
                    return
            elif any(value <= 0 for value in absorption_inputs.values()):
                st.error("Both attenuation coefficients and path lengths must be positive.")
                return
        if calculate_clicked and calculate_set and (
            set_inputs["total_thickness_mm"] <= 0
            or set_inputs["scintillator_density_g_cm3"] <= 0
            or set_inputs["matrix_density_g_cm3"] <= 0
        ):
            st.error("Enter positive layer thickness and component densities.")
            return
        if not calculate_clicked and (
            not cached_result
            or cached_result["request_fingerprint"] != request_fingerprint
        ):
            st.caption(
                "Upload a reference and enter the required values, then calculate "
                "the apparent yield."
            )
            return

        if calculate_clicked:
            try:
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    reference_path = temporary_path / Path(reference_upload.name).name
                    reference_path.write_bytes(reference_upload.getvalue())
                    reference_df = read_spectrum(reference_path, spectrum_axis)
                    sample_spectrum = [
                        (float(row.wavelength_nm), float(row.intensity))
                        for row in spectrum_df.itertuples()
                    ]
                    reference_spectrum = [
                        (float(row.wavelength_nm), float(row.intensity))
                        for row in reference_df.itertuples()
                    ]
                    background_paths: dict[str, str] = {}
                    for label, background_upload, spectrum in (
                        ("sample", sample_background_upload, sample_spectrum),
                        ("reference", reference_background_upload, reference_spectrum),
                    ):
                        if background_upload:
                            background_path = temporary_path / (
                                f"{label}_{Path(background_upload.name).name}"
                            )
                            background_path.write_bytes(background_upload.getvalue())
                            background_df = read_spectrum(
                                background_path, spectrum_axis
                            )
                            background_spectrum = [
                                (float(row.wavelength_nm), float(row.intensity))
                                for row in background_df.itertuples()
                            ]
                            corrected_spectrum = subtract_background(
                                spectrum, background_spectrum
                            )
                            if label == "sample":
                                sample_spectrum = corrected_spectrum
                            else:
                                reference_spectrum = corrected_spectrum
                            background_paths[label] = Path(
                                background_upload.name
                            ).name

                    detector_correction = None
                    if detector_upload:
                        detector_path = temporary_path / Path(detector_upload.name).name
                        detector_path.write_bytes(detector_upload.getvalue())
                        detector_df = read_spectrum(detector_path, "wavelength_nm")
                        detector_correction = [
                            (float(row.wavelength_nm), float(row.intensity))
                            for row in detector_df.itertuples()
                        ]

                    results = calculate_apparent_light_yield(
                        sample_spectrum,
                        reference_spectrum,
                        reference_light_yield,
                        detector_correction,
                    )
                    if apply_absorption_correction:
                        if absorption_method == "Enter sample/reference attenuation fractions":
                            sample_fraction = absorption_inputs["sample_fraction"]
                            reference_fraction = absorption_inputs["reference_fraction"]
                        else:
                            sample_fraction = calculate_monoenergetic_absorption_fraction(
                                absorption_inputs["sample_mu"],
                                absorption_inputs["sample_path_mm"],
                            )
                            reference_fraction = calculate_monoenergetic_absorption_fraction(
                                absorption_inputs["reference_mu"],
                                absorption_inputs["reference_path_mm"],
                            )
                        results["sample_xray_attenuation_fraction"] = sample_fraction
                        results["reference_xray_attenuation_fraction"] = reference_fraction
                        results["xray_attenuation_corrected_light_yield_photons_per_mev"] = (
                            calculate_absorption_corrected_light_yield(
                                results["apparent_light_yield_photons_per_mev"],
                                sample_fraction,
                                reference_fraction,
                            )
                        )
                    if calculate_set:
                        results["scintillator_equivalent_thickness_mm"] = (
                            calculate_scintillator_equivalent_thickness(
                                set_inputs["total_thickness_mm"],
                                set_inputs["mass_fraction"],
                                set_inputs["scintillator_density_g_cm3"],
                                set_inputs["matrix_density_g_cm3"],
                            )
                        )

                    report = {
                        "method": (
                            "Reference-based apparent light yield from the ratio of "
                            "detector-weighted integrated emission spectra."
                        ),
                        "sample_spectrum_file": Path(uploaded_file.name).name,
                        "reference": {
                            "spectrum_file": Path(reference_upload.name).name,
                            "light_yield_photons_per_MeV": reference_light_yield,
                            "value_status": reference_status,
                            "source": reference_source.strip(),
                        },
                        "background_files": background_paths,
                        "detector_response": {
                            "handling": detector_mode,
                            "correction_file": (
                                Path(detector_upload.name).name if detector_upload else None
                            ),
                        },
                        "absorption_correction": {
                            "method": absorption_method,
                            "inputs": absorption_inputs,
                        },
                        "scintillator_equivalent_thickness_inputs": set_inputs,
                        "results": results,
                        "assumptions_and_limits": [
                            "Sample and reference must have comparable X-ray fluence, collection geometry, and acquisition settings.",
                            "When acquisition settings differ, normalize sample and reference signals to equivalent X-ray exposure before analysis.",
                            "The detector correction is a multiplicative function applied to both spectra; do not apply it a second time to pre-corrected data.",
                            "Spectra are linearly interpolated and integrated only over their shared wavelength range; calibration extrapolation is not allowed.",
                            "X-ray attenuation correction normalizes both sample and reference attenuation fractions and does not correct optical transport or independently determine deposited energy.",
                            "The monochromatic attenuation estimate assumes a homogeneous layer and a normal-incidence primary beam; use energy-weighted attenuation fractions for a polychromatic beam.",
                            "The scintillator-equivalent thickness uses the mass-fraction/density equation in the Supporting Information.",
                            "The yield inherits absolute units only when the reference light yield is traceable.",
                        ],
                    }
                    report_bytes = (
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                    ).encode("utf-8")
                    st.session_state[result_state_key] = {
                        "request_fingerprint": request_fingerprint,
                        "results": results,
                        "report_bytes": report_bytes,
                    }
                    cached_result = st.session_state[result_state_key]
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not calculate apparent XEOL light yield: {exc}")
                return

        results = cached_result["results"]
        if (
            reference_status == "Certified/traceable absolute value"
            and detector_mode != "No correction available (uncorrected ratio)"
        ):
            st.success("Apparent light yield calculated on the supplied reference scale.")
            yield_label = "Apparent light yield (photons/MeV)"
        elif reference_status == "Certified/traceable absolute value":
            st.warning(
                "The reference is traceable, but no detector-response correction "
                "was applied; this apparent yield may be wavelength-response biased."
            )
            yield_label = "Uncorrected apparent yield (reference scale)"
        else:
            st.warning(
                "The reference is approximate or relative; this result is not an "
                "absolute light yield."
            )
            yield_label = "Reference-scaled apparent light yield (not absolute)"
        st.metric(
            yield_label,
            f"{results['apparent_light_yield_photons_per_mev']:.6g}",
        )
        signal_columns = st.columns(3)
        signal_columns[0].metric(
            "Integrated sample signal",
            f"{results['sample_integrated_signal']:.6g}",
        )
        signal_columns[1].metric(
            "Integrated reference signal",
            f"{results['reference_integrated_signal']:.6g}",
        )
        signal_columns[2].metric(
            "Shared wavelength range",
            f"{results['wavelength_min_nm']:g}-{results['wavelength_max_nm']:g} nm",
        )
        corrected_yield = results.get(
            "xray_attenuation_corrected_light_yield_photons_per_mev"
        )
        if corrected_yield is not None:
            st.metric(
                "X-ray-attenuation-corrected yield (photons/MeV)",
                f"{corrected_yield:.6g}",
            )
        set_thickness = results.get("scintillator_equivalent_thickness_mm")
        if set_thickness is not None:
            st.metric(
                "Scintillator-equivalent thickness d",
                f"{set_thickness:.6g} mm",
            )
        st.download_button(
            "Download apparent light-yield report (JSON)",
            cached_result["report_bytes"],
            file_name="apparent_xeol_light_yield_report.json",
            mime="application/json",
            key=f"apparent_report_download_{spectrum_identity}",
        )


def render_absolute_xeol_yield(
    uploaded_file,
    spectrum_df,
    spectrum_axis,
    spectrum_identity,
):
    """Render and calculate XEOL yield when a full absolute calibration is supplied."""
    with st.expander("Absolute XEOL light yield (photons/keV deposited)"):
        st.warning(
            "The QEPro wavelength certificate calibrates wavelength only. "
            "Absolute yield also requires a separate complete-chain radiometric "
            "response (W/nm per count/s), a defensible collection efficiency, "
            "and a deposited-energy estimate. Do not use the wavelength "
            "certificate as the radiometric response."
        )
        st.write(
            "This uses the spectral-response approach adapted from de Haas, "
            "Dorenbos, and van Eijk (2005); it does not apply their PMT-specific "
            "quantum-efficiency or reflectivity corrections."
        )
        response_upload = st.file_uploader(
            "Absolute spectral response CSV",
            type=["csv", "txt"],
            key=f"xeol_response_{spectrum_identity}",
            help=(
                "Required columns: wavelength_nm and "
                "watts_per_nm_per_count_rate. Values must calibrate the full "
                "optical chain, not just wavelength or relative response."
            ),
        )
        background_upload = st.file_uploader(
            "Optional matching background spectrum",
            type=["txt", "csv", "dat"],
            key=f"xeol_background_{spectrum_identity}",
            help="Use the same acquisition settings; the background must cover the measured wavelength range.",
        )

        signal_basis = st.selectbox(
            "QEPro signal units",
            ["Counts per integration", "Counts per second"],
            key=f"xeol_signal_basis_{spectrum_identity}",
            help="Choose the units in the uploaded spectrum; scan averaging is not multiplied into exposure time automatically.",
        )
        integration_time_s = None
        if signal_basis == "Counts per integration":
            integration_time_s = st.number_input(
                "QEPro integration time (s)",
                min_value=0.0,
                value=0.0,
                key=f"xeol_integration_{spectrum_identity}",
                help="Enter the integration duration corresponding to the uploaded counts.",
            )
        exposure_time_s = st.number_input(
            "Actual X-ray beam-on duration for this spectrum (s)",
            min_value=0.0,
            value=0.0,
            key=f"xeol_exposure_{spectrum_identity}",
            help="Enter actual exposure duration; do not infer it by multiplying integration time by scans.",
        )

        column_left, column_right = st.columns(2)
        incident_flux = column_left.number_input(
            "Incident X-ray flux (photons/s)",
            min_value=0.0,
            value=0.0,
            format="%.6g",
            key=f"xeol_flux_{spectrum_identity}",
        )
        xray_energy_kev = column_right.number_input(
            "X-ray photon energy (keV)",
            min_value=0.0,
            value=0.0,
            format="%.6g",
            key=f"xeol_xray_energy_{spectrum_identity}",
        )
        deposited_fraction = st.number_input(
            "Estimated fraction of incident X-ray energy deposited in sample",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            format="%.6f",
            key=f"xeol_deposited_fraction_{spectrum_identity}",
            help="Supply a justified energy-deposition fraction; this is not inferred from primary-beam attenuation.",
        )

        collection_mode = st.selectbox(
            "Collection-efficiency correction",
            ["Single wavelength-independent fraction", "Wavelength-dependent CSV"],
            key=f"xeol_collection_mode_{spectrum_identity}",
        )
        collection_efficiency: float | list[tuple[float, float]] = 0.0
        collection_upload = None
        if collection_mode == "Single wavelength-independent fraction":
            collection_efficiency = st.number_input(
                "Fraction of emitted photons represented at calibrated detector (0 < η ≤ 1)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                format="%.6f",
                key=f"xeol_collection_scalar_{spectrum_identity}",
                help="This is separate from the response calibration; avoid correcting the same optical losses twice.",
            )
        else:
            collection_upload = st.file_uploader(
                "Collection efficiency CSV",
                type=["csv", "txt"],
                key=f"xeol_collection_curve_{spectrum_identity}",
                help="Required columns: wavelength_nm and collection_efficiency (fractions from 0 to 1).",
            )

        st.markdown("**Sample and calibration metadata**")
        metadata_left, metadata_right = st.columns(2)
        sample_composition = metadata_left.text_input(
            "Sample composition",
            key=f"xeol_composition_{spectrum_identity}",
        )
        sample_thickness_mm = metadata_right.number_input(
            "Sample thickness (mm)",
            min_value=0.0,
            value=0.0,
            key=f"xeol_thickness_{spectrum_identity}",
        )
        collection_geometry = st.text_input(
            "Collection geometry (describe sample, fiber, optics, and path)",
            key=f"xeol_geometry_{spectrum_identity}",
        )
        radiometric_calibration_id = st.text_input(
            "Absolute radiometric calibration identifier",
            key=f"xeol_radiometric_id_{spectrum_identity}",
            help="Identifier for the separate absolute response calibration, not the QEPro wavelength certificate.",
        )
        wavelength_calibration_id = st.text_input(
            "QEPro wavelength-certificate identifier (optional)",
            key=f"xeol_wavelength_id_{spectrum_identity}",
        )
        include_uncertainties = st.checkbox(
            "Enter relative standard uncertainties (%)",
            value=False,
            key=f"xeol_include_uncertainties_{spectrum_identity}",
            help="If enabled, enter all five values for independent uncertainties in response, collection efficiency, flux, X-ray energy, and deposited fraction.",
        )
        uncertainty_inputs: dict[str, str] = {}
        if include_uncertainties:
            st.caption(
                "Enter one-standard-deviation relative uncertainties as percentages."
            )
            uncertainty_columns = st.columns(5)
            uncertainty_labels = {
                "absolute spectral response": "Response (%)",
                "collection efficiency": "Collection (%)",
                "incident flux": "Flux (%)",
                "X-ray energy": "X-ray energy (%)",
                "deposited fraction": "Deposited fraction (%)",
            }
            for column, (name, label) in zip(
                uncertainty_columns, uncertainty_labels.items()
            ):
                uncertainty_inputs[name] = column.text_input(
                    label,
                    key=f"xeol_uncertainty_{name}_{spectrum_identity}",
                    help="Required when uncertainty calculation is selected.",
                )

        calculate_clicked = st.button(
            "Calculate absolute light yield",
            key=f"xeol_calculate_{spectrum_identity}",
            type="primary",
        )
        calculation_inputs = {
            "signal_basis": signal_basis,
            "integration_time_s": integration_time_s,
            "exposure_time_s": exposure_time_s,
            "incident_flux": incident_flux,
            "xray_energy_kev": xray_energy_kev,
            "deposited_fraction": deposited_fraction,
            "collection_efficiency": collection_efficiency,
            "collection_mode": collection_mode,
            "sample_composition": sample_composition,
            "sample_thickness_mm": sample_thickness_mm,
            "collection_geometry": collection_geometry,
            "radiometric_calibration_id": radiometric_calibration_id,
            "wavelength_calibration_id": wavelength_calibration_id,
            "uncertainty_inputs": uncertainty_inputs,
        }
        request_hash = hashlib.sha256(
            json.dumps(calculation_inputs, sort_keys=True).encode("utf-8")
        )
        for uploaded_item in (
            uploaded_file,
            response_upload,
            background_upload,
            collection_upload,
        ):
            if uploaded_item is not None:
                request_hash.update(uploaded_item.getvalue())
        request_fingerprint = request_hash.hexdigest()
        result_state_key = f"xeol_result_{spectrum_identity}"
        cached_result = st.session_state.get(result_state_key)

        required_text = {
            "sample composition": sample_composition,
            "collection geometry": collection_geometry,
            "absolute radiometric calibration identifier": radiometric_calibration_id,
        }
        missing_text = [name for name, value in required_text.items() if not value.strip()]
        if calculate_clicked and missing_text:
            st.error("Enter the following required metadata: " + ", ".join(missing_text) + ".")
            return
        if calculate_clicked and not response_upload:
            st.error(
                "Upload an absolute spectral-response table. A wavelength-only "
                "QEPro certificate is not sufficient."
            )
            return
        positive_inputs = {
            "actual beam-on duration": exposure_time_s,
            "incident X-ray flux": incident_flux,
            "X-ray photon energy": xray_energy_kev,
            "deposited fraction": deposited_fraction,
            "sample thickness": sample_thickness_mm,
        }
        if integration_time_s is not None:
            positive_inputs["integration time"] = integration_time_s
        if calculate_clicked and (
            any(value <= 0 for value in positive_inputs.values())
            or deposited_fraction > 1
        ):
            st.error("All acquisition and sample inputs must be greater than zero.")
            return
        if calculate_clicked and collection_mode == "Wavelength-dependent CSV" and not collection_upload:
            st.error("Upload the wavelength-dependent collection-efficiency CSV.")
            return
        uncertainties_percent: dict[str, float | None] = {
            "absolute spectral response": None,
            "collection efficiency": None,
            "incident flux": None,
            "X-ray energy": None,
            "deposited fraction": None,
        }
        if calculate_clicked and include_uncertainties:
            try:
                uncertainties_percent = {
                    name: float(value.strip())
                    for name, value in uncertainty_inputs.items()
                }
                if any(
                    not np.isfinite(value) or value < 0
                    for value in uncertainties_percent.values()
                    if value is not None
                ):
                    raise ValueError
            except ValueError:
                st.error(
                    "Enter a finite, non-negative percentage for each of the "
                    "five uncertainty inputs."
                )
                return
        if not calculate_clicked and (
            not cached_result
            or cached_result["request_fingerprint"] != request_fingerprint
        ):
            st.caption(
                "No current absolute result. Supply valid calibration and "
                "measurement inputs, then calculate the yield."
            )
            return

        if calculate_clicked:
            try:
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    spectrum_path = temporary_path / Path(uploaded_file.name).name
                    response_path = temporary_path / Path(response_upload.name).name
                    spectrum_path.write_bytes(uploaded_file.getvalue())
                    response_path.write_bytes(response_upload.getvalue())
                    spectrum = [
                        (float(wavelength), float(signal))
                        for wavelength, signal in zip(
                            spectrum_df["wavelength_nm"],
                            spectrum_df["intensity"],
                        )
                    ]
                    background_path = None
                    if background_upload:
                        background_path = temporary_path / Path(background_upload.name).name
                        background_path.write_bytes(background_upload.getvalue())
                        background = read_xeol_spectrum(background_path, spectrum_axis)
                        background_values = [
                            (float(row.wavelength_nm), float(row.intensity))
                            for row in background.itertuples()
                        ]
                        spectrum = subtract_background(spectrum, background_values)

                    response_curve = read_spectral_response(response_path)
                    spectral_power = correct_spectral_response(
                        spectrum,
                        response_curve,
                        integration_time_s,
                        (
                            "counts_per_integration"
                            if signal_basis == "Counts per integration"
                            else "counts_per_second"
                        ),
                    )
                    collection_description = ""
                    if collection_upload:
                        collection_path = temporary_path / Path(collection_upload.name).name
                        collection_path.write_bytes(collection_upload.getvalue())
                        collection_efficiency = read_collection_efficiency(collection_path)
                        collection_description = (
                            f"Wavelength-dependent calibration: {collection_upload.name}"
                        )
                    else:
                        if collection_efficiency <= 0 or collection_efficiency > 1:
                            raise ValueError(
                                "Collection efficiency must be greater than 0 and at most 1."
                            )
                        collection_description = (
                            f"Scalar fraction: {collection_efficiency:g}"
                        )

                    deposited_energy_kev = calculate_deposited_xray_energy(
                        incident_flux,
                        exposure_time_s,
                        xray_energy_kev,
                        deposited_fraction,
                    )
                    results = calculate_absolute_light_yield(
                        spectral_power,
                        exposure_time_s,
                        collection_efficiency,
                        deposited_energy_kev,
                    )
                    report_path = temporary_path / "xeol_light_yield_report.json"
                    report = generate_xeol_report(
                        spectrum_path,
                        response_path,
                        report_path,
                        {
                            "composition": sample_composition.strip(),
                            "thickness_mm": sample_thickness_mm,
                            "collection_geometry": collection_geometry.strip(),
                            "radiometric_calibration_identifier": radiometric_calibration_id.strip(),
                            "wavelength_calibration_identifier": (
                                wavelength_calibration_id.strip() or "not provided"
                            ),
                        },
                        {
                            "incident_flux_photons_s": incident_flux,
                            "photon_energy_keV": xray_energy_kev,
                            "actual_exposure_time_s": exposure_time_s,
                            "deposited_fraction_user_supplied": deposited_fraction,
                            "estimated_deposited_energy_keV": deposited_energy_kev,
                        },
                        results,
                        uncertainties_percent,
                        background_path,
                        collection_description,
                        {
                            "signal_basis": signal_basis,
                            "integration_time_s": (
                                integration_time_s
                                if integration_time_s is not None
                                else "not applicable"
                            ),
                        },
                        wavelength_range_nm=(
                            min(wavelength for wavelength, _ in spectrum),
                            max(wavelength for wavelength, _ in spectrum),
                        ),
                    )
                    st.session_state[result_state_key] = {
                        "request_fingerprint": request_fingerprint,
                        "results": results,
                        "deposited_energy_kev": deposited_energy_kev,
                        "wavelength_range": [
                            min(wavelength for wavelength, _ in spectrum),
                            max(wavelength for wavelength, _ in spectrum),
                        ],
                        "report_bytes": report_path.read_bytes(),
                        "uncertainty": report["uncertainty"],
                    }
                    cached_result = st.session_state[result_state_key]
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not calculate absolute XEOL light yield: {exc}")
                return

        results = cached_result["results"]
        deposited_energy_kev = cached_result["deposited_energy_kev"]
        st.success("Absolute light-yield estimate calculated.")
        st.metric(
            "Absolute XEOL light yield",
            f"{results['light_yield_photons_per_keV']:.6g} photons/keV deposited",
        )
        result_columns = st.columns(3)
        result_columns[0].metric(
            "Estimated emitted photons",
            f"{results['emitted_photons']:.6g}",
        )
        result_columns[1].metric(
            "Estimated deposited X-ray energy",
            f"{deposited_energy_kev:.6g} keV",
        )
        result_columns[2].metric(
            "Measured wavelength range",
            (
                f"{cached_result['wavelength_range'][0]:g}-"
                f"{cached_result['wavelength_range'][1]:g} nm"
            ),
        )
        uncertainty = cached_result["uncertainty"]
        uncertainty_value = uncertainty[
            "light_yield_standard_uncertainty_photons_per_keV"
        ]
        if uncertainty_value is None:
            st.info(
                "Combined uncertainty not calculated. Missing relative standard "
                "uncertainties: "
                + ", ".join(uncertainty["uncertainties_not_provided"])
                + "."
            )
        else:
            st.metric(
                "Combined standard uncertainty",
                f"±{uncertainty_value:.6g} photons/keV",
            )
        st.download_button(
            "Download XEOL light-yield report (JSON)",
            cached_result["report_bytes"],
            file_name="xeol_light_yield_report.json",
            mime="application/json",
            key=f"xeol_report_download_{spectrum_identity}",
        )


def main():
    st.set_page_config(page_title="LuminescenceLab: Rare Earth Spectral Analysis", layout="wide")
    st.title("LuminescenceLab: Rare Earth Spectral Analysis")
    st.write(
        "Detect emission peaks and compare their transition energies with the "
        "available Carnall energy-level database."
    )

    database_path = Path(__file__).resolve().with_name("carnall_energy_database.json")
    try:
        with database_path.open("r", encoding="utf-8") as database_file:
            database = json.load(database_file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load the Carnall energy database: {exc}")
        st.stop()

    uploaded_files = st.sidebar.file_uploader(
        "Upload spectrum(s)", type=["txt", "csv", "dat"], accept_multiple_files=True
    )
    axis_options = {
        "Automatic detection": "auto",
        "Wavelength (nm)": "wavelength_nm",
        "Wavenumber (cm⁻¹)": "wavenumber_cm1",
        "Photon energy (eV)": "energy_ev",
    }
    axis_label = st.sidebar.selectbox("Spectral position axis", list(axis_options))
    if len(uploaded_files) > 1:
        st.subheader("Spectra Comparison")
        st.write(
            "Each uploaded spectrum is plotted independently using its original "
            "intensity values. No spectra are combined, averaged, or normalized."
        )
        comparison_mode = st.radio(
            "Plot layout",
            ["Overlay", "Stacked"],
            horizontal=True,
        )
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                spectra = []
                comparison_identity = hashlib.sha256()
                for index, uploaded_file in enumerate(uploaded_files):
                    file_data = uploaded_file.getvalue()
                    comparison_identity.update(uploaded_file.name.encode("utf-8"))
                    comparison_identity.update(file_data)
                    spectrum_path = Path(temporary_directory) / (
                        f"scan_{index}{Path(uploaded_file.name).suffix}"
                    )
                    spectrum_path.write_bytes(file_data)
                    spectra.append((
                        uploaded_file.name,
                        read_spectrum(spectrum_path, axis_options[axis_label]),
                    ))
                if len({spectrum.attrs.get("axis") for _, spectrum in spectra}) > 1:
                    raise ValueError(
                        "The spectra were detected with different x-axis units. "
                        "Choose one spectral coordinate for all files before plotting."
                    )
                all_x = np.concatenate([
                    spectrum["x"].to_numpy(float) for _, spectrum in spectra
                ])
                all_intensity = np.concatenate([
                    spectrum["intensity"].to_numpy(float)
                    for _, spectrum in spectra
                ])
                intensity_default_min = float(all_intensity.min())
                intensity_default_max = float(all_intensity.max())
                if intensity_default_min == intensity_default_max:
                    intensity_padding = max(abs(intensity_default_min) * 0.01, 0.5)
                    intensity_default_min -= intensity_padding
                    intensity_default_max += intensity_padding
                controls_key = comparison_identity.hexdigest()[:12]
                st.sidebar.header("Comparison Plot Controls")
                dark_background = st.sidebar.toggle(
                    "Dark plot background",
                    value=True,
                    key=f"comparison_dark_background_{controls_key}",
                )
                show_intensity_background = st.sidebar.toggle(
                    "Show intensity background colormap",
                    value=False,
                    key=f"comparison_intensity_background_{controls_key}",
                )
                background_colormap = "inferno"
                if show_intensity_background:
                    background_colormap = BACKGROUND_COLORMAP_OPTIONS[
                        st.sidebar.selectbox(
                            "Intensity background colormap",
                            options=list(BACKGROUND_COLORMAP_OPTIONS),
                            index=0,
                            key=f"comparison_colormap_{controls_key}",
                        )
                    ]
                show_scan_labels = st.sidebar.checkbox(
                    "Show scan labels / legend",
                    value=True,
                    key=f"comparison_labels_{controls_key}",
                )
                comparison_linewidth = st.sidebar.slider(
                    "Spectrum line width",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.8,
                    step=0.1,
                    key=f"comparison_linewidth_{controls_key}",
                )
                st.sidebar.subheader("Editable Scan Labels")
                scan_labels = []
                for index, (filename, _) in enumerate(spectra):
                    scan_label = st.sidebar.text_input(
                        f"Label for {filename}",
                        value=Path(filename).stem,
                        key=f"comparison_scan_label_{controls_key}_{index}",
                    ).strip()
                    if not scan_label:
                        st.sidebar.error(
                            f"Enter a label for scan {filename}, or remove the file."
                        )
                        st.stop()
                    scan_labels.append(scan_label)
                spectra = [
                    (scan_labels[index], spectrum)
                    for index, (_, spectrum) in enumerate(spectra)
                ]
                cie_rows = []
                valid_cie_rows = []
                for label, spectrum in spectra:
                    try:
                        coordinates = calculate_cie_coordinates(
                            spectrum["wavelength_nm"].to_numpy(float),
                            spectrum["intensity"].to_numpy(float),
                        )
                    except ValueError as exc:
                        cie_rows.append(
                            {
                                "No": "",
                                "Spectrum": label,
                                "Status": f"Unavailable: {exc}",
                            }
                        )
                    else:
                        cie_row = {
                            "No": len(valid_cie_rows) + 1,
                            "Spectrum": label,
                            "Status": "Calculated",
                            **coordinates,
                        }
                        cie_rows.append(cie_row)
                        valid_cie_rows.append(cie_row)
                x_limit_columns = st.sidebar.columns(2)
                x_min = x_limit_columns[0].number_input(
                    "E_min",
                    value=float(all_x.min()),
                    format="%.6f",
                    key=f"comparison_xmin_{controls_key}",
                    help="Lower plot limit in the selected spectral coordinate.",
                )
                x_max = x_limit_columns[1].number_input(
                    "E_max",
                    value=float(all_x.max()),
                    format="%.6f",
                    key=f"comparison_xmax_{controls_key}",
                    help="Upper plot limit in the selected spectral coordinate.",
                )
                intensity_limit_columns = st.sidebar.columns(2)
                intensity_min = intensity_limit_columns[0].number_input(
                    "Intensity_min",
                    value=intensity_default_min,
                    format="%.6f",
                    key=f"comparison_intensity_min_{controls_key}",
                )
                intensity_max = intensity_limit_columns[1].number_input(
                    "Intensity_max",
                    value=intensity_default_max,
                    format="%.6f",
                    key=f"comparison_intensity_max_{controls_key}",
                )
                st.sidebar.subheader("Comparison Font Sizes")
                axis_label_fontsize = st.sidebar.slider(
                    "Axis label font size",
                    min_value=8,
                    max_value=24,
                    value=16,
                    key=f"comparison_axis_label_fontsize_{controls_key}",
                )
                tick_fontsize = st.sidebar.slider(
                    "Tick font size",
                    min_value=6,
                    max_value=20,
                    value=12,
                    key=f"comparison_tick_fontsize_{controls_key}",
                )
                legend_fontsize = st.sidebar.slider(
                    "Label / legend font size",
                    min_value=6,
                    max_value=20,
                    value=10,
                    key=f"comparison_legend_fontsize_{controls_key}",
                )
                if x_min >= x_max:
                    st.sidebar.error("E_min must be less than E_max.")
                    st.stop()
                if intensity_min >= intensity_max:
                    st.sidebar.error(
                        "Intensity_min must be less than Intensity_max."
                    )
                    st.stop()
                comparison_plot_path = Path(temporary_directory) / "spectrum_comparison.png"
                plot_spectrum_comparison(
                    spectra,
                    comparison_mode,
                    comparison_plot_path,
                    spectra[0][1].attrs.get(
                        "axis_label", axis_options[axis_label]
                    ),
                    linewidth=comparison_linewidth,
                    show_scan_labels=show_scan_labels,
                    x_axis_limits=(x_min, x_max),
                    y_axis_limits=(intensity_min, intensity_max),
                    x_label_fontsize=axis_label_fontsize,
                    y_label_fontsize=axis_label_fontsize,
                    x_tick_fontsize=tick_fontsize,
                    y_tick_fontsize=tick_fontsize,
                    legend_fontsize=legend_fontsize,
                    dark_background=dark_background,
                    show_intensity_background=show_intensity_background,
                    background_colormap=background_colormap,
                )
                comparison_plot_bytes = comparison_plot_path.read_bytes()
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not plot the uploaded scans: {exc}")
            st.stop()
        st.image(comparison_plot_bytes, use_container_width=True)
        st.download_button(
            "Download comparison plot",
            comparison_plot_bytes,
            file_name="spectrum_comparison.png",
            mime="image/png",
        )
        with st.expander("CIE 1931 2° chromaticity coordinates", expanded=True):
            st.caption(
                "Each scan is integrated independently against the CIE 1931 2° "
                "standard-observer color-matching functions. Accurate color "
                "coordinates require wavelength-dependent correction for the "
                "detector and optical response."
            )
            cie_table = pd.DataFrame(cie_rows)
            st.dataframe(cie_table, hide_index=True, use_container_width=True)
            st.download_button(
                "Download CIE coordinates (CSV)",
                data=cie_table.to_csv(index=False).encode("utf-8"),
                file_name="cie_1931_coordinates.csv",
                mime="text/csv",
                key=f"cie_coordinates_comparison_download_{controls_key}",
            )
            if valid_cie_rows:
                try:
                    chromaticity_image = create_cie_chromaticity_image(
                        valid_cie_rows
                    )
                except ValueError as exc:
                    st.error(f"Could not create the CIE diagram: {exc}")
                else:
                    st.download_button(
                        "Download numbered CIE diagram (PNG)",
                        data=chromaticity_image,
                        file_name="cie_1931_chromaticity_diagram.png",
                        mime="image/png",
                        key=f"cie_diagram_comparison_download_{controls_key}",
                    )
        st.stop()
    uploaded_file = uploaded_files[0] if uploaded_files else None

    st.sidebar.header("Peak Detection")
    smooth_window = st.sidebar.slider(
        "Smoothing window (odd number of points)",
        min_value=2,
        max_value=51,
        value=9,
        step=2,
    )
    relative_prominence = st.sidebar.checkbox(
        "Use relative peak prominence", value=True
    )
    prominence = st.sidebar.number_input(
        "Peak prominence",
        min_value=0.0,
        value=0.005,
        step=0.001,
        format="%.6f",
        help=(
            "As a fraction of the signal range when relative prominence is "
            "enabled; otherwise in intensity units."
        ),
    )
    distance_points = st.sidebar.number_input(
        "Minimum peak separation (points)",
        min_value=1,
        value=5,
        step=1,
    )
    min_height_enabled = st.sidebar.checkbox(
        "Set minimum peak intensity",
        value=False,
    )
    min_height = None
    if min_height_enabled:
        min_height = st.sidebar.number_input(
            "Minimum intensity",
            value=0.0,
            format="%.6f",
        )

    st.sidebar.header("Transition Matching")
    ion_options = ["All ions", "Auto-detect from filename"] + list(database.keys())
    ion_choice = st.sidebar.selectbox(
        "Lanthanide ion",
        options=ion_options,
        index=1,
    )
    tolerance_nm = st.sidebar.number_input(
        "Wavelength tolerance (nm)",
        min_value=0.01,
        max_value=100.0,
        value=3.0,
        step=0.5,
    )
    use_cm1_tolerance = st.sidebar.checkbox(
        "Use transition-energy tolerance (cm⁻¹)",
        value=False,
    )
    tolerance_cm1 = None
    if use_cm1_tolerance:
        tolerance_cm1 = st.sidebar.number_input(
            "Energy tolerance (cm⁻¹)",
            min_value=1.0,
            max_value=10000.0,
            value=200.0,
            step=10.0,
        )
    max_matches = st.sidebar.slider(
        "Maximum candidates per peak",
        min_value=1,
        max_value=10,
        value=5,
    )
    st.sidebar.header("Plot Appearance")
    dark_background = st.sidebar.toggle(
        "Dark plot background",
        value=True,
        key="single_dark_background",
    )
    show_intensity_background = st.sidebar.toggle(
        "Show intensity background colormap",
        value=False,
        key="single_intensity_background",
    )
    peak_label_fontsize = st.sidebar.slider(
        "Peak label font size",
        min_value=6,
        max_value=20,
        value=10,
    )
    axis_label_fontsize = st.sidebar.slider(
        "Axis label font size",
        min_value=8,
        max_value=24,
        value=16,
    )
    tick_label_fontsize = st.sidebar.slider(
        "Tick label font size",
        min_value=6,
        max_value=20,
        value=12,
    )
    spectrum_linewidth = st.sidebar.slider(
        "Spectrum line width",
        min_value=0.5,
        max_value=5.0,
        value=1.8,
        step=0.1,
    )
    smoothed_signal_color = st.sidebar.color_picker(
        "Smoothed signal color",
        value="#ff5a1f",
    )
    show_peak_labels = st.sidebar.checkbox(
        "Show detected peaks (markers and labels)",
        value=True,
    )
    background_colormap = "inferno"
    if show_intensity_background:
        background_colormap = BACKGROUND_COLORMAP_OPTIONS[
            st.sidebar.selectbox(
                "Intensity background colormap",
                options=list(BACKGROUND_COLORMAP_OPTIONS),
                index=0,
                help=(
                    "Emission spectra: inferno, magma, plasma, viridis, turbo, hot, "
                    "and cividis. Inferno and magma work especially well on dark "
                    "backgrounds. Scientific-comparison maps are better suited to "
                    "positive/negative residuals than ordinary intensity data. "
                    "Intensity-map palettes suit 2D distributions. Rainbow and HSV "
                    "can create misleading visual boundaries; twilight maps suit "
                    "cyclic data."
                ),
            )
        ]
    if uploaded_file is None:
        st.info("Upload one or more TXT, CSV or DAT spectra using the sidebar.")
        st.markdown("#### Supported input")
        st.write(
            "A text file with two columns: spectral position and intensity. "
            "Whitespace-, comma- and semicolon-separated files are supported."
        )
        st.markdown("#### Analysis workflow")
        st.write(
            "Upload spectrum → detect peaks → match transitions → "
            "inspect the annotated plot and download the results."
        )
        st.stop()

    ion_filter = None
    if ion_choice == "Auto-detect from filename":
        ion_filter = infer_ion_from_filename(uploaded_file.name, database)
    elif ion_choice != "All ions":
        ion_filter = ion_choice

    try:
        uploaded_data = uploaded_file.getvalue()
        spectrum_identity = hashlib.sha256(uploaded_data).hexdigest()[:12]
        cie_coordinates = None
        cie_error = None
        with tempfile.TemporaryDirectory() as temporary_directory:
            spectrum_path = Path(temporary_directory) / (
                f"uploaded_spectrum{Path(uploaded_file.name).suffix}"
            )
            spectrum_path.write_bytes(uploaded_data)
            df = read_spectrum(spectrum_path, axis_options[axis_label])
            try:
                cie_coordinates = calculate_cie_coordinates(
                    df["wavelength_nm"].to_numpy(float),
                    df["intensity"].to_numpy(float),
                )
            except ValueError as exc:
                cie_error = str(exc)
            x_axis_limits = None
            if df.attrs.get("axis") == "wavelength_nm":
                wavelength_columns = st.sidebar.columns(2)
                wavelength_min = wavelength_columns[0].number_input(
                    "E_min (nm)",
                    min_value=0.000001,
                    value=float(df["x"].min()),
                    step=1.0,
                    format="%.6f",
                    key=f"wavelength_emin_{spectrum_identity}",
                    help="Lower wavelength bound for the displayed spectrum.",
                )
                wavelength_max = wavelength_columns[1].number_input(
                    "E_max (nm)",
                    min_value=0.000001,
                    value=float(df["x"].max()),
                    step=1.0,
                    format="%.6f",
                    key=f"wavelength_emax_{spectrum_identity}",
                    help="Upper wavelength bound for the displayed spectrum.",
                )
                if wavelength_min >= wavelength_max:
                    st.sidebar.error("E_min must be less than E_max.")
                    st.stop()
                x_axis_limits = (wavelength_min, wavelength_max)
            intensity_columns = st.sidebar.columns(2)
            intensity_min = intensity_columns[0].number_input(
                "Intensity_min",
                value=float(df["intensity"].min()),
                format="%.6f",
                key=f"intensity_min_{spectrum_identity}",
                help="Lower intensity bound for the displayed spectrum.",
            )
            intensity_max = intensity_columns[1].number_input(
                "Intensity_max",
                value=float(df["intensity"].max()),
                format="%.6f",
                key=f"intensity_max_{spectrum_identity}",
                help="Upper intensity bound for the displayed spectrum.",
            )
            if intensity_min > intensity_max:
                st.sidebar.error("Intensity_min cannot exceed Intensity_max.")
                st.stop()
            y_axis_limits = (intensity_min, intensity_max)
            applied_prominence = (
                prominence * (df["intensity"].max() - df["intensity"].min())
                if relative_prominence
                else prominence
            )
            smoothed, peaks = detect_peaks(
                df,
                smooth_window=smooth_window,
                prominence=applied_prominence,
                distance_points=distance_points,
                min_height=min_height,
            )
            matches, catalog = match_peaks(
                peaks,
                database,
                max_matches=max_matches,
                ion_filter=ion_filter,
                tolerance_nm=tolerance_nm,
                tolerance_cm1=tolerance_cm1,
            )
            plot_path = Path(temporary_directory) / "annotated_spectrum.png"
            plot_results(
                df,
                smoothed,
                peaks,
                matches,
                plot_path,
                df.attrs.get("axis_label", axis_options[axis_label]),
                peak_label_fontsize=peak_label_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                background_colormap=background_colormap,
                spectrum_linewidth=spectrum_linewidth,
                smoothed_signal_color=smoothed_signal_color,
                tick_label_fontsize=tick_label_fontsize,
                show_peak_labels=show_peak_labels,
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits,
                dark_background=dark_background,
                show_intensity_background=show_intensity_background,
            )
            plot_bytes = plot_path.read_bytes()
            publication_plot_path = Path(temporary_directory) / "publication_spectrum.png"
            plot_results(
                df,
                smoothed,
                peaks,
                matches,
                publication_plot_path,
                df.attrs.get("axis_label", axis_options[axis_label]),
                peak_label_fontsize=peak_label_fontsize,
                axis_label_fontsize=axis_label_fontsize,
                background_colormap=background_colormap,
                spectrum_linewidth=spectrum_linewidth,
                smoothed_signal_color=smoothed_signal_color,
                tick_label_fontsize=tick_label_fontsize,
                show_peak_labels=show_peak_labels,
                x_axis_limits=x_axis_limits,
                y_axis_limits=y_axis_limits,
                show_title=False,
                show_provisional_note=False,
                dark_background=dark_background,
                show_intensity_background=show_intensity_background,
            )
            publication_plot_bytes = publication_plot_path.read_bytes()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not analyze the uploaded spectrum: {exc}")
        st.stop()

    with st.expander("CIE 1931 2° chromaticity coordinates", expanded=True):
        st.caption(
            "Calculated by integrating the measured emission spectrum against "
            "the CIE 1931 2° standard-observer color-matching functions. "
            "Accurate color coordinates require wavelength-dependent correction "
            "for the detector and optical response."
        )
        if cie_coordinates is None:
            st.warning(f"Could not calculate CIE coordinates: {cie_error}")
        else:
            coordinate_columns = st.columns(3)
            coordinate_columns[0].metric("CIE x", f"{cie_coordinates['cie_x']:.6f}")
            coordinate_columns[1].metric("CIE y", f"{cie_coordinates['cie_y']:.6f}")
            coordinate_columns[2].metric("CIE z", f"{cie_coordinates['cie_z']:.6f}")
            st.caption(
                "Relative tristimulus values (XYZ): "
                f"{cie_coordinates['tristimulus_X']:.6g}, "
                f"{cie_coordinates['tristimulus_Y']:.6g}, "
                f"{cie_coordinates['tristimulus_Z']:.6g}; "
                f"integration range: "
                f"{cie_coordinates['wavelength_min_nm']:.3f}–"
                f"{cie_coordinates['wavelength_max_nm']:.3f} nm."
            )
            st.download_button(
                "Download CIE coordinates (CSV)",
                data=pd.DataFrame(
                    [
                        {
                            "No": 1,
                            "Spectrum": Path(uploaded_file.name).name,
                            **cie_coordinates,
                        }
                    ]
                ).to_csv(index=False).encode("utf-8"),
                file_name="cie_1931_coordinates.csv",
                mime="text/csv",
                key=f"cie_coordinates_download_{spectrum_identity}",
            )
            try:
                chromaticity_image = create_cie_chromaticity_image(
                    [
                        {
                            "No": 1,
                            "Spectrum": Path(uploaded_file.name).name,
                            **cie_coordinates,
                        }
                    ]
                )
            except ValueError as exc:
                st.error(f"Could not create the CIE diagram: {exc}")
            else:
                st.download_button(
                    "Download numbered CIE diagram (PNG)",
                    data=chromaticity_image,
                    file_name="cie_1931_chromaticity_diagram.png",
                    mime="image/png",
                    key=f"cie_diagram_download_{spectrum_identity}",
                )

    render_apparent_xeol_yield(
        uploaded_file, df, axis_options[axis_label], spectrum_identity
    )
    render_absolute_xeol_yield(
        uploaded_file, df, axis_options[axis_label], spectrum_identity
    )

    matched_peak_count = (
        int(matches.loc[matches["status"].eq("CANDIDATE_MATCH"), "peak_id"].nunique())
        if not matches.empty
        else 0
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Data points", f"{len(df):,}")
    summary_columns[1].metric("Detected peaks", f"{len(peaks):,}")
    summary_columns[2].metric("Peaks with candidates", f"{matched_peak_count:,}")

    st.image(plot_bytes, caption="Spectrum plot", use_container_width=True)
    st.caption("Integration limits (edit per peak)")
    st.write(
        "Limits default to each peak's FWHM boundaries. Area uses the raw, "
        "un-baseline-corrected intensity and trapezoidal integration."
    )
    table_columns = st.columns([0.7, 1.0, 1.5, 1.5, 1.2])
    for column, label in zip(
        table_columns,
        (
            "Peak_id",
            "X_peak",
            "Integration start (X1)",
            "Integration end (X2)",
            "Integrated area",
        ),
    ):
        column.markdown(f"**{label}**")

    peaks = peaks.copy()
    integration_editor_key = (
        f"peak_integration_{spectrum_identity}_{smooth_window}_"
        f"{applied_prominence:.8g}_{distance_points}_{min_height}"
    )
    if peaks.empty:
        st.write("No peaks detected.")
    else:
        for index, peak in peaks.iterrows():
            row_columns = st.columns([0.7, 1.0, 1.5, 1.5, 1.2])
            row_columns[0].write(int(peak["peak_id"]))
            row_columns[1].write(f"{peak['x_peak']:.6f}")
            integration_x1 = row_columns[2].number_input(
                "Integration start (X1)",
                value=float(peak["integration_x1"]),
                format="%.6f",
                label_visibility="collapsed",
                key=f"{integration_editor_key}_{int(peak['peak_id'])}_x1",
            )
            integration_x2 = row_columns[3].number_input(
                "Integration end (X2)",
                value=float(peak["integration_x2"]),
                format="%.6f",
                label_visibility="collapsed",
                key=f"{integration_editor_key}_{int(peak['peak_id'])}_x2",
            )
            try:
                integrated_area = integrate_spectrum_interval(
                    df, integration_x1, integration_x2
                )
            except (TypeError, ValueError) as exc:
                st.error(f"Could not calculate peak integrated area: {exc}")
                st.stop()
            row_columns[4].write(f"{integrated_area:.6g}")
            peaks.at[index, "integration_x1"] = integration_x1
            peaks.at[index, "integration_x2"] = integration_x2
            peaks.at[index, "integrated_area"] = integrated_area

    show_judd_ofelt = uploaded_file is not None and ion_filter == "Eu3+"
    if show_judd_ofelt:
        st.subheader("Judd–Ofelt parameters (Eu³⁺)")
        st.write(
            "Edit the transition wavelengths, integrated peak areas, refractive "
            "index, and lifetime below. The fields start with the supplied default "
            "values and are not automatically copied from the per-peak integration "
            "table. Verify that entered areas and units match the JO model."
        )
        jo_defaults = JuddOfeltInputs()
        jo_widget_key = f"judd_ofelt_{spectrum_identity}"

        st.caption("Electronic transition positions (nm)")
        wavelength_columns = st.columns(4)
        transition_wavelengths = {
            "Transition5D0_7F1": wavelength_columns[0].number_input(
                "⁵D₀ → ⁷F₁",
                min_value=0.000001,
                value=jo_defaults.Transition5D0_7F1,
                format="%.10g",
                key=f"{jo_widget_key}_Transition5D0_7F1",
            ),
            "Transition5D0_7F2": wavelength_columns[1].number_input(
                "⁵D₀ → ⁷F₂",
                min_value=0.000001,
                value=jo_defaults.Transition5D0_7F2,
                format="%.10g",
                key=f"{jo_widget_key}_Transition5D0_7F2",
            ),
            "Transition5D0_7F4": wavelength_columns[2].number_input(
                "⁵D₀ → ⁷F₄",
                min_value=0.000001,
                value=jo_defaults.Transition5D0_7F4,
                format="%.10g",
                key=f"{jo_widget_key}_Transition5D0_7F4",
            ),
            "Transition5D0_7F6": wavelength_columns[3].number_input(
                "⁵D₀ → ⁷F₆",
                min_value=0.000001,
                value=jo_defaults.Transition5D0_7F6,
                format="%.10g",
                key=f"{jo_widget_key}_Transition5D0_7F6_810nm",
            ),
        }

        st.caption("Integrated transition peak areas (S)")
        area_columns = st.columns(5)
        transition_areas = {
            "S5D0_7F0": area_columns[0].number_input(
                "S(⁵D₀ → ⁷F₀)",
                min_value=0.0,
                value=jo_defaults.S5D0_7F0,
                format="%.10g",
                key=f"{jo_widget_key}_S5D0_7F0",
            ),
            "S5D0_7F1": area_columns[1].number_input(
                "S(⁵D₀ → ⁷F₁)",
                min_value=0.000000000001,
                value=jo_defaults.S5D0_7F1,
                format="%.10g",
                key=f"{jo_widget_key}_S5D0_7F1",
            ),
            "S5D0_7F2": area_columns[2].number_input(
                "S(⁵D₀ → ⁷F₂)",
                min_value=0.000000000001,
                value=jo_defaults.S5D0_7F2,
                format="%.10g",
                key=f"{jo_widget_key}_S5D0_7F2",
            ),
            "S5D0_7F4": area_columns[3].number_input(
                "S(⁵D₀ → ⁷F₄)",
                min_value=0.0,
                value=jo_defaults.S5D0_7F4,
                format="%.10g",
                key=f"{jo_widget_key}_S5D0_7F4",
            ),
            "S5D0_7F6": area_columns[4].number_input(
                "S(⁵D₀ → ⁷F₆)",
                min_value=0.0,
                value=jo_defaults.S5D0_7F6,
                format="%.10g",
                key=f"{jo_widget_key}_S5D0_7F6",
            ),
        }

        st.caption("Refractive index and lifetime")
        material_columns = st.columns(2)
        refractive_index = material_columns[0].number_input(
            "Refractive index (n)",
            min_value=0.000000000001,
            value=jo_defaults.n,
            format="%.10g",
            key=f"{jo_widget_key}_n",
        )
        lifetime_s = material_columns[1].number_input(
            "Lifetime (τ, s)",
            min_value=0.000000000001,
            value=jo_defaults.tau_s,
            format="%.10g",
            key=f"{jo_widget_key}_tau_s",
        )

        jo_inputs = JuddOfeltInputs(
            **transition_wavelengths,
            **transition_areas,
            n=refractive_index,
            tau_s=lifetime_s,
        )
        jo_results = calculate_judd_ofelt(jo_inputs)
        st.subheader("Result of calculated Judd-Ofelt (JO) parameters")
        result_columns = st.columns(4)
        for column, label, scaled_key, raw_key in (
            (result_columns[0], "Ω₂ × 10²⁰", "rho2", "Omega2"),
            (result_columns[1], "Ω₄ × 10²⁰", "rho4", "Omega4"),
            (result_columns[2], "Ω₆ × 10²⁰", "rho6", "Omega6"),
        ):
            column.metric(label, f"{jo_results[scaled_key]:.10g}")
            column.caption(f"Unscaled Ω = {jo_results[raw_key]:.10g}")
        result_columns[3].metric(
            "Intrinsic quantum yield (%)",
            f"{jo_results['eta_percent']:.10g}%",
        )

        with st.expander("View all parameters calculated by the JO theory"):
            result_names = {
                "sigma01_cm^-1": "σ(⁵D₀→⁷F₁) (cm⁻¹)",
                "sigma02_cm^-1": "σ(⁵D₀→⁷F₂) (cm⁻¹)",
                "sigma04_cm^-1": "σ(⁵D₀→⁷F₄) (cm⁻¹)",
                "sigma06_cm^-1": "σ(⁵D₀→⁷F₆) (cm⁻¹)",
                "chi": "Refractive-index correction (χ)",
                "A01": "A(⁵D₀→⁷F₁)",
                "A02": "A(⁵D₀→⁷F₂)",
                "A04": "A(⁵D₀→⁷F₄)",
                "A06": "A(⁵D₀→⁷F₆)",
                "Omega2": "Unscaled Ω₂",
                "Omega4": "Unscaled Ω₄",
                "Omega6": "Unscaled Ω₆",
                "rho2": "Ω₂ × 10²⁰",
                "rho4": "Ω₄ × 10²⁰",
                "rho6": "Ω₆ × 10²⁰",
                "R02": "S(⁵D₀→⁷F₀) / S(⁵D₀→⁷F₂)",
                "Arad": "Radiative transition probability (A_rad)",
                "Atotal": "Total transition probability (A_total)",
                "Anr": "Non-radiative transition probability (A_nr)",
                "eta_percent": "Intrinsic quantum yield (%)",
                "nH2O": "nH₂O",
            }
            st.dataframe(
                pd.DataFrame(
                    {
                        "Quantity": [result_names[name] for name in jo_results],
                        "Value": [value for value in jo_results.values()],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    downloads = st.columns(5)
    downloads[0].download_button(
        "Detected peaks CSV",
        peaks.to_csv(index=False),
        file_name="detected_peaks.csv",
        mime="text/csv",
    )
    downloads[1].download_button(
        "Transition matches CSV",
        matches.to_csv(index=False),
        file_name="peak_transition_matches.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "Transition catalog CSV",
        catalog.to_csv(index=False),
        file_name="database_transition_catalog.csv",
        mime="text/csv",
    )
    downloads[3].download_button(
        "Converted spectrum CSV",
        df.to_csv(index=False),
        file_name="spectrum_with_converted_energy.csv",
        mime="text/csv",
    )
    downloads[4].download_button(
        "Save publication ready plot",
        publication_plot_bytes,
        file_name="annotated_spectrum_300dpi.png",
        mime="image/png",
        help="Exported at 300 dpi with a 7 x 4.5 inch figure size.",
    )

    st.divider()
    st.markdown(
        "If you use LuminescenceLab in your research, please cite: "
        "Khan, L. U. et al. (2026). “Lighting up materials with X-rays: "
        "Advanced X-ray excited optical luminescence probe at the "
        "BM08-XAFS/XRF beamline, SESAME.” *IUCrJ*, **13**. "
        "[View the manuscript](https://journals.iucr.org/m/issues/2026/06/00/ro5048/index.html)\n\n"
        "For work involving calculation of Judd-Ofelt (JO) parameters, also cite: "
        "Khan, L. U. et al. (2014). “Red-green emitting and superparamagnetic "
        "nanomarkers containing Fe₃O₄ functionalized with calixarene and rare "
        "earth complexes.” *Inorganic Chemistry*, **53**(24), 12902–12910. "
        "[https://doi.org/10.1021/ic5018856](https://doi.org/10.1021/ic5018856)"
    )


if __name__ == "__main__":
    main()
