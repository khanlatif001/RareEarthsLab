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
        df["photon_cm1"] = 1e7 / x
        label = "Wavelength (nm)"
    elif axis == "wavenumber_cm1":
        if np.any(x <= 0): raise ValueError("Wavenumbers must be positive.")
        df["photon_cm1"] = x
        df["wavelength_nm"] = 1e7 / x
        label = "Wavenumber (cm$^{-1}$)"
    elif axis == "energy_ev":
        if np.any(x <= 0): raise ValueError("Photon energies must be positive.")
        df["photon_cm1"] = x * 8065.544005
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
    match = re.fullmatch(r"(\d+)([A-Za-z])(\d+)", str(state))
    if not match:
        return str(state)
    multiplicity, term, angular_momentum = match.groups()
    return rf"$^{{{multiplicity}}}{term}_{{{angular_momentum}}}$"


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
    widths = peak_widths(ys, peaks, rel_height=0.5)[0] if len(peaks) else np.array([])
    # Robust estimate of high-frequency residual noise; uncertainty is approximate,
    # intended as a diagnostic rather than a formal fit-derived confidence interval.
    residual = y - ys
    noise_sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    x_axis = df["x"].to_numpy(float)
    dx = float(np.median(np.diff(x_axis)))
    out = []
    for i, p in enumerate(peaks):
        row = {
            "peak_id": int(i+1),
            "x_peak": float(df.iloc[p]["x"]),
            "wavelength_nm": float(df.iloc[p]["wavelength_nm"]),
            "photon_cm1": float(df.iloc[p]["photon_cm1"]),
            "intensity_raw": float(y[p]),
            "intensity_smoothed": float(ys[p]),
            "prominence": float(props["prominences"][i]),
            "width_points_fwhm_approx": float(widths[i]) if len(widths) else np.nan,
            "width_x_fwhm_approx": float(widths[i] * dx) if len(widths) else np.nan,
            "x_uncertainty_approx": float(max(dx / 2.0, (widths[i] * dx) * noise_sigma /
                                               max(float(props["prominences"][i]), 1e-12)))
                                    if len(widths) else dx / 2.0,
            "noise_sigma_robust": float(noise_sigma),
            "peak_index": int(p),
        }
        out.append(row)
    return ys, pd.DataFrame(out)


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
        catalog["delta_cm1"] = catalog["transition_cm1"] - float(peak["photon_cm1"])
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
                "observed_photon_cm1": float(peak["photon_cm1"]),
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
                 show_peak_labels=True, x_axis_limits=None):
    x = df["x"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#050505")
    intensity_range = float(np.ptp(y))
    display_max = float(np.max(y))
    if intensity_range > 0 and display_max > 0:
        intensity_map = np.tile((y - np.min(y)) / intensity_range, (32, 1))
        x_edges = np.empty(x.size + 1)
        x_edges[1:-1] = (x[:-1] + x[1:]) / 2
        x_edges[0] = x[0] - (x[1] - x[0]) / 2
        x_edges[-1] = x[-1] + (x[-1] - x[-2]) / 2
        y_edges = np.linspace(0, display_max * 1.08, intensity_map.shape[0] + 1)
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
    if not peaks.empty:
        ax.scatter(peaks["x_peak"], peaks["intensity_smoothed"], s=24, zorder=4,
                   marker="o", color="white", edgecolors="#ff5a1f", label="Detected peaks")
        if show_peak_labels:
            # Label each detected peak; top candidate shown only if a plausible match exists.
            for _, p in peaks.iterrows():
                subset = matches[(matches["peak_id"] == p["peak_id"]) &
                                 matches.get("status", pd.Series(dtype=str)).isin(
                                     ["CANDIDATE_MATCH"])]
                if not subset.empty and pd.notna(subset.iloc[0].get("ion")):
                    m = subset.iloc[0]
                    label = (f'{m["ion"]} {format_state(m["upper_state"])} → '
                             f'{format_state(m["lower_state"])}\n'
                             f'ref {m["reference_wavelength_nm"]:.1f} nm '
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
    ax.set_xlabel(axis_label, fontsize=axis_label_fontsize)
    ax.set_ylabel("Intensity (a.u.)", fontsize=axis_label_fontsize)
    title = ("Emission spectrum with peak and transition labels"
             if show_peak_labels else "Emission spectrum with detected peaks")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white", labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(color="white", alpha=0.15)
    legend = ax.legend(facecolor="#202020", edgecolor="white", labelcolor="white")
    transcription_status = matches.get("transcription_status", pd.Series(dtype=str)).fillna("").astype(str)
    if transcription_status.str.contains("provisional", case=False).any():
        fig.text(0.5, 0.015, "Provisional database transcription; verify before scientific use.",
                 ha="center", fontsize=8, color="white")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    st.set_page_config(page_title="Rare Earth Spectral Peak Analysis", layout="wide")
    st.title("Rare Earth Spectral Peak Analysis")
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

    uploaded_file = st.sidebar.file_uploader(
        "Upload a spectrum", type=["txt", "csv", "dat"]
    )
    axis_options = {
        "Automatic detection": "auto",
        "Wavelength (nm)": "wavelength_nm",
        "Wavenumber (cm⁻¹)": "wavenumber_cm1",
        "Photon energy (eV)": "energy_ev",
    }
    axis_label = st.sidebar.selectbox("Spectral position axis", list(axis_options))

    st.sidebar.header("Peak detection")
    smooth_window = st.sidebar.slider(
        "Smoothing window (odd number of points)",
        min_value=5,
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

    st.sidebar.header("Transition matching")
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
    st.sidebar.header("Plot appearance")
    peak_label_fontsize = st.sidebar.slider(
        "Peak label font size",
        min_value=8,
        max_value=20,
        value=10,
    )
    axis_label_fontsize = st.sidebar.slider(
        "X/Y axis label font size",
        min_value=10,
        max_value=24,
        value=16,
    )
    tick_label_fontsize = st.sidebar.slider(
        "X/Y tick label font size",
        min_value=8,
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
        "Show peak transition and position labels",
        value=True,
    )
    colormap_options = {
        "Inferno": "inferno",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Magma": "magma",
        "Cividis": "cividis",
        "Turbo": "turbo",
    }
    background_colormap = colormap_options[
        st.sidebar.selectbox(
            "Intensity background colormap",
            options=list(colormap_options),
            index=0,
        )
    ]
    x_axis_limits = None
    if axis_options[axis_label] == "wavelength_nm":
        set_wavelength_limits = st.sidebar.checkbox(
            "Set wavelength range",
            value=False,
        )
        if set_wavelength_limits:
            wavelength_columns = st.sidebar.columns(2)
            wavelength_min = wavelength_columns[0].number_input(
                "Minimum wavelength (nm)",
                min_value=0.000001,
                value=350.0,
                step=1.0,
            )
            wavelength_max = wavelength_columns[1].number_input(
                "Maximum wavelength (nm)",
                min_value=0.000001,
                value=900.0,
                step=1.0,
            )
            if wavelength_min >= wavelength_max:
                st.sidebar.error(
                    "Minimum wavelength must be less than maximum wavelength."
                )
                st.stop()
            x_axis_limits = (wavelength_min, wavelength_max)

    if uploaded_file is None:
        st.info("Upload a TXT, CSV or DAT spectrum using the sidebar to begin.")
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
        with tempfile.TemporaryDirectory() as temporary_directory:
            spectrum_path = Path(temporary_directory) / (
                f"uploaded_spectrum{Path(uploaded_file.name).suffix}"
            )
            spectrum_path.write_bytes(uploaded_file.getvalue())
            df = read_spectrum(spectrum_path, axis_options[axis_label])
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
            )
            plot_bytes = plot_path.read_bytes()
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
        "Publication-ready plot PNG",
        plot_bytes,
        file_name="annotated_spectrum_300dpi.png",
        mime="image/png",
        help="Exported at 300 dpi with a 7 x 4.5 inch figure size.",
    )


if __name__ == "__main__":
    main()
