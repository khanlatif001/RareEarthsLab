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


def read_spectrum(path, axis="auto"):
    """Read a two-column whitespace/comma separated TXT/CSV spectrum."""
    df = pd.read_csv(path, sep=r"[\s,;]+", comment="#", header=None,
                     engine="python", usecols=[0, 1], names=["x", "intensity"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
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
                 show_provisional_note=True, y_axis_limits=None):
    x = df["x"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#050505")
    intensity_range = float(np.ptp(y))
    display_max = float(np.max(y))
    plot_y_limits = y_axis_limits
    if y_axis_limits is not None and y_axis_limits[0] == y_axis_limits[1]:
        padding = max(abs(y_axis_limits[0]) * 0.01, 0.5)
        plot_y_limits = (
            y_axis_limits[0] - padding,
            y_axis_limits[1] + padding,
        )
    if intensity_range > 0 and display_max > 0:
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
        colorbar.ax.tick_params(colors="white")
    ax.set_facecolor("#050505")
    ax.plot(x, y, lw=spectrum_linewidth, alpha=0.9, color="#ffd447",
            label="Experimental spectrum")
    ax.plot(x, smoothed, lw=spectrum_linewidth, color=smoothed_signal_color,
            label="Smoothed signal")
    if show_peak_labels and not peaks.empty:
        ax.scatter(peaks["x_peak"], peaks["intensity_smoothed"], s=24, zorder=4,
                   marker="o", color="white", edgecolors="#ff5a1f", label="Detected peaks")
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
                        fontsize=peak_label_fontsize, color="white", rotation=90,
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
        ax.set_title(title, color="white")
    ax.tick_params(colors="white", labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(color="white", alpha=0.15)
    if show_peak_labels:
        ax.legend(facecolor="#202020", edgecolor="white", labelcolor="white")
    transcription_status = matches.get("transcription_status", pd.Series(dtype=str)).fillna("").astype(str)
    if (show_provisional_note and
            transcription_status.str.contains("provisional", case=False).any()):
        fig.text(0.5, 0.015, "Provisional database transcription; verify before scientific use.",
                 ha="center", fontsize=8, color="white")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_spectrum_comparison(spectra, mode, output_png, axis_label, linewidth=1.8):
    """Plot independent scans as an overlay or in vertically stacked axes."""
    if not spectra:
        raise ValueError("Upload at least one spectrum to plot.")
    if mode not in {"Overlay", "Stacked"}:
        raise ValueError(f"Unsupported comparison layout {mode!r}.")

    colors = plt.get_cmap("tab20", len(spectra))
    if mode == "Overlay":
        fig, ax = plt.subplots(figsize=(9, 5.5))
        axes = [ax]
    else:
        fig, axes_array = plt.subplots(
            len(spectra),
            1,
            figsize=(9, max(4.5, 2.5 * len(spectra))),
            sharex=True,
            squeeze=False,
        )
        axes = list(axes_array[:, 0])

    fig.patch.set_facecolor("#050505")
    for index, (name, spectrum) in enumerate(spectra):
        ax = axes[0] if mode == "Overlay" else axes[index]
        ax.set_facecolor("#050505")
        ax.plot(
            spectrum["x"],
            spectrum["intensity"],
            color=colors(index),
            linewidth=linewidth,
            label=name,
        )
        ax.tick_params(colors="white")
        ax.grid(color="white", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.yaxis.label.set_color("white")
        if mode == "Stacked":
            ax.set_ylabel(name, color="white")

    if mode == "Overlay":
        axes[0].set_ylabel("Intensity (a.u.)", color="white")
        axes[0].legend(
            facecolor="#202020",
            edgecolor="white",
            labelcolor="white",
            loc="best",
        )
    else:
        axes[-1].set_xlabel(axis_label, color="white")
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
    if mode == "Overlay":
        axes[0].set_xlabel(axis_label, color="white")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


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
        st.subheader("Scan comparison")
        st.write(
            "Each uploaded scan is plotted independently using its original "
            "intensity values. No scans are combined, averaged, or normalized."
        )
        comparison_mode = st.radio(
            "Plot layout",
            ["Overlay", "Stacked"],
            horizontal=True,
        )
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                spectra = []
                for index, uploaded_file in enumerate(uploaded_files):
                    spectrum_path = Path(temporary_directory) / (
                        f"scan_{index}{Path(uploaded_file.name).suffix}"
                    )
                    spectrum_path.write_bytes(uploaded_file.getvalue())
                    spectra.append((
                        uploaded_file.name,
                        read_spectrum(spectrum_path, axis_options[axis_label]),
                    ))
                if len({spectrum.attrs.get("axis") for _, spectrum in spectra}) > 1:
                    raise ValueError(
                        "The scans were detected with different x-axis units. "
                        "Choose one spectral coordinate for all files before plotting."
                    )
                comparison_plot_path = Path(temporary_directory) / "scan_comparison.png"
                plot_spectrum_comparison(
                    spectra,
                    comparison_mode,
                    comparison_plot_path,
                    spectra[0][1].attrs.get(
                        "axis_label", axis_options[axis_label]
                    ),
                    linewidth=1.8,
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
    colormap_options = {
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
    background_colormap = colormap_options[
        st.sidebar.selectbox(
            "Intensity background colormap",
            options=list(colormap_options),
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
        with tempfile.TemporaryDirectory() as temporary_directory:
            spectrum_path = Path(temporary_directory) / (
                f"uploaded_spectrum{Path(uploaded_file.name).suffix}"
            )
            spectrum_path.write_bytes(uploaded_data)
            df = read_spectrum(spectrum_path, axis_options[axis_label])
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
            )
            publication_plot_bytes = publication_plot_path.read_bytes()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not analyze the uploaded spectrum: {exc}")
        st.stop()

    matched_peak_count = (
        int(matches.loc[matches["status"].eq("CANDIDATE_MATCH"), "peak_id"].nunique())
        if not matches.empty
        else 0
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Data points", f"{len(df):,}")
    summary_columns[1].metric("Detected peaks", f"{len(peaks):,}")
    summary_columns[2].metric("Peaks with candidates", f"{matched_peak_count:,}")

    st.image(plot_bytes, caption="Annotated spectrum", use_container_width=True)
    st.subheader("Detected peaks")
    if peaks.empty:
        st.dataframe(peaks, use_container_width=True)
    else:
        st.caption("Integration limits (edit per peak)")
        st.write(
            "Integration limits start at each peak's FWHM boundaries. Edit x1 and "
            "x2 for individual peaks; area uses the raw, un-baseline-corrected "
            "intensity and trapezoidal integration."
        )
        bounds_columns = ["peak_id", "x_peak", "integration_x1", "integration_x2"]
        integration_editor_key = (
            f"peak_integration_{spectrum_identity}_{smooth_window}_"
            f"{applied_prominence:.8g}_{distance_points}_{min_height}"
        )
        edited_bounds = st.data_editor(
            peaks[bounds_columns],
            column_config={
                "integration_x1": st.column_config.NumberColumn(
                    "Integration start (x1)", format="%.6f"
                ),
                "integration_x2": st.column_config.NumberColumn(
                    "Integration end (x2)", format="%.6f"
                ),
            },
            disabled=["peak_id", "x_peak"],
            hide_index=True,
            use_container_width=True,
            key=integration_editor_key,
        )
        try:
            peaks = peaks.copy()
            peaks["integration_x1"] = edited_bounds["integration_x1"]
            peaks["integration_x2"] = edited_bounds["integration_x2"]
            peaks["integrated_area"] = [
                integrate_spectrum_interval(
                    df, float(row["integration_x1"]), float(row["integration_x2"])
                )
                for _, row in peaks.iterrows()
            ]
        except (TypeError, ValueError) as exc:
            st.error(f"Could not calculate peak integrated area: {exc}")
            st.stop()
        st.dataframe(peaks, use_container_width=True)
    st.subheader("Transition candidates")
    st.dataframe(matches, use_container_width=True)

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
        "[View the manuscript](https://journals.iucr.org/m/issues/2026/06/00/ro5048/index.html)"
    )


if __name__ == "__main__":
    main()
