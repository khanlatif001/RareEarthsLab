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
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
import matplotlib.pyplot as plt


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


def plot_results(df, smoothed, peaks, matches, output_png, axis_label):
    x = df["x"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(15, 7))
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
                      cmap="inferno", alpha=1.0, zorder=0)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno"), ax=ax, pad=0.015)
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Min", "Max"])
        colorbar.ax.tick_params(colors="white")
    ax.set_facecolor("#050505")
    ax.plot(x, y, lw=1.8, alpha=0.9, color="#ffd447", label="Experimental spectrum")
    ax.plot(x, smoothed, lw=2.2, color="#ff5a1f", label="Smoothed signal")
    if not peaks.empty:
        ax.scatter(peaks["x_peak"], peaks["intensity_smoothed"], s=24, zorder=4,
                   marker="o", color="white", edgecolors="#ff5a1f", label="Detected peaks")
        # Label each detected peak; top candidate shown only if a plausible match exists.
        for peak_number, (_, p) in enumerate(peaks.iterrows()):
            subset = matches[(matches["peak_id"] == p["peak_id"]) &
                             matches.get("status", pd.Series(dtype=str)).isin(
                                 ["CANDIDATE_MATCH"])]
            if not subset.empty and pd.notna(subset.iloc[0].get("ion")):
                m = subset.iloc[0]
                label = (f'{m["ion"]} {format_state(m["upper_state"])} → '
                         f'{format_state(m["lower_state"])}\n'
                         f'ref {m["reference_wavelength_nm"]:.1f} nm '
                         f'(Δ {m["delta_reference_minus_observed_nm"]:+.1f} nm)')
                label_color = "white"
            else:
                label = f'Peak {int(p["peak_id"])}: no database match'
                label_color = "white"
            label += f'\n{p["wavelength_nm"]:.1f} nm'
            label_above = p["intensity_smoothed"] <= display_max * 0.55
            offset = (0, 8) if label_above else (0, -8)
            vertical_alignment = "bottom" if label_above else "top"
            ax.annotate(label, (p["x_peak"], p["intensity_smoothed"]),
                        xytext=offset, textcoords="offset points",
                        fontsize=8, color=label_color, rotation=90,
                        ha="center", va=vertical_alignment)
    if x.min() < 350 and x.max() > 900 and "Wavelength" in axis_label:
        ax.set_xlim(350, 900)
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Emission spectrum with peak and transition labels", color="white")
    ax.tick_params(colors="white")
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
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", nargs="?", help="Input TXT/CSV spectrum; omit to browse for a file")
    parser.add_argument("--database", default=None,
                        help="Carnall database JSON file")
    parser.add_argument("--axis", choices=["auto", "wavelength_nm", "wavenumber_cm1", "energy_ev"],
                        default="auto", help="First column quantity (default: infer from its scale)")
    parser.add_argument("--output", default=None, help="Output directory (default: <spectrum>_results)")
    parser.add_argument("--smooth-window", type=int, default= 9,
                        help="Odd Savitzky-Golay window length in points")
    parser.add_argument("--prominence", type=float, default=0.005,
                        help="Peak prominence in intensity units (or range fraction if --relative-prominence)")
    parser.add_argument("--relative-prominence", action="store_true",
                        help="Interpret prominence as a fraction of full signal range")
    parser.add_argument("--absolute-prominence", dest="relative_prominence", action="store_false",
                        help="Interpret prominence in intensity units")
    parser.add_argument("--distance", type=int, default=5,
                        help="Minimum separation between peaks in data points")
    parser.add_argument("--min-height", type=float, default=None,
                        help="Optional minimum smoothed intensity")
    parser.add_argument("--tolerance-nm", type=float, default=3.0,
                        help="Wavelength match tolerance in nm (default: 3)")
    parser.add_argument("--tolerance-cm1", "--tolerance", dest="tolerance_cm1",
                        type=float, default=None,
                        help="Legacy transition-energy tolerance in cm^-1; overrides --tolerance-nm")
    parser.add_argument("--max-matches", type=int, default=5,
                        help="Maximum candidate transitions retained per peak")
    parser.add_argument("--ion", default="auto",
                        help='Database ion filter, e.g. "Eu3+" or "Pr3+,Nd3+"; default: infer from filename, or search all ions')
    parser.set_defaults(relative_prominence=True)
    args = parser.parse_args()

    spectrum_path = Path(args.spectrum) if args.spectrum else choose_spectrum()
    database_path = Path(args.database) if args.database else (
        Path(__file__).resolve().with_name("carnall_energy_database.json"))
    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    ion_filter = infer_ion_from_filename(spectrum_path, database) if args.ion == "auto" else args.ion
    outdir = Path(args.output) if args.output else spectrum_path.with_name(f"{spectrum_path.stem}_results")
    outdir.mkdir(parents=True, exist_ok=True)
    df = read_spectrum(spectrum_path, args.axis)
    prominence = args.prominence * (df["intensity"].max() - df["intensity"].min()) if args.relative_prominence else args.prominence
    smoothed, peaks = detect_peaks(df, smooth_window=args.smooth_window,
                                   prominence=prominence, distance_points=args.distance,
                                   min_height=args.min_height)
    matches, catalog = match_peaks(peaks, database, max_matches=args.max_matches,
                                   ion_filter=ion_filter, tolerance_nm=args.tolerance_nm,
                                   tolerance_cm1=args.tolerance_cm1)
    df.to_csv(outdir / "spectrum_with_converted_energy.csv", index=False)
    peaks.to_csv(outdir / "detected_peaks.csv", index=False)
    matches.to_csv(outdir / "peak_transition_matches.csv", index=False)
    catalog.to_csv(outdir / "database_transition_catalog.csv", index=False)
    pd.DataFrame({"x": df["x"], "intensity": df["intensity"], "smoothed": smoothed}).to_csv(
        outdir / "smoothed_spectrum.csv", index=False)
    plot_results(df, smoothed, peaks, matches, outdir / "annotated_spectrum.png",
                 df.attrs.get("axis_label", args.axis))
    summary = {
        "spectrum": str(spectrum_path), "axis": df.attrs.get("axis", args.axis),
        "ion_filter": ion_filter, "data_points": int(len(df)),
        "match_tolerance": ({"value": args.tolerance_cm1, "unit": "cm^-1"}
                            if args.tolerance_cm1 is not None else
                            {"value": args.tolerance_nm, "unit": "nm"}),
        "detected_peaks": int(len(peaks)), "database_ions": int(len(database)),
        "ions_with_levels": {ion: len(info.get("levels", [])) for ion, info in database.items()},
        "ions_with_explicit_transitions": {
            ion: len(info.get("transitions", [])) for ion, info in database.items()
        },
        "transition_candidates_in_catalog": int(len(catalog)),
        "database_matched_peaks": (int(
            matches.loc[matches["status"].eq("CANDIDATE_MATCH"), "peak_id"].nunique()
        ) if not matches.empty else 0),
        "note": "Assignments are tentative. Database transitions are matched by wavelength within the configured nm tolerance; provisional OCR transcriptions require source verification."
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\\nResults saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
