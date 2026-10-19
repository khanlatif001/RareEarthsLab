# RareEarthsLab

**A free web application for rare earth emission spectroscopy and element peaks /
transition identification.**

[![Launch RareEarthsLab](https://img.shields.io/badge/Launch-RareEarthsLab-168B8B?style=for-the-badge)](https://rareearthslab-jxhqdrpyr3mtukkjbdaugf.streamlit.app/)

RareEarthsLab helps researchers inspect experimental emission spectra, detect
peaks, and compare them with available reference energy levels for trivalent
lanthanide ions. It combines peak detection, spectral conversion, candidate
transition matching, plot customization, and downloadable results in a
browser-based interface.

> **Scientific note:** Matches are candidate assignments, not definitive
> identifications. The included energy-level database is a partial, evolving
> transcription and contains provisional entries. Verify source data and
> assignments against the original literature before scientific use.

## Features

### Spectrum analysis

- Upload two-column TXT, CSV, or DAT files. Whitespace-, comma-, and
  semicolon-separated input is supported.
- Use wavelength (nm), wavenumber (cm⁻¹), or photon energy (eV) as the spectral
  coordinate, or let the application infer the axis from the data.
- Smooth the signal with a Savitzky–Golay filter before detecting peaks.
- Tune peak prominence, minimum point separation, smoothing window, and
  optional minimum intensity.
- Convert input positions to wavelength and photon wavenumber for comparison.

### Candidate transition matching

- Compare peak photon energies with transitions derived from the available
  reference energy levels.
- Match using a wavelength tolerance or an optional transition-energy
  tolerance (cm⁻¹).
- Select an ion, infer an ion from the uploaded filename where possible, or
  search across the available ions.
- Inspect candidate states, reference wavelengths, energy differences, and
  available source and transcription details.

The quality and coverage of matches depend on experimental calibration,
peak-detection settings, and reference-data completeness.

### Plot customization

Use the sidebar to adjust:

- Peak-label font size, axis-label font size, and tick-label font size.
- Line width shared by the experimental and smoothed signals.
- Smoothed-signal color.
- Visibility of detected peak markers and their transition/position labels.
- The intensity-background colormap.
- **Emin (nm)** and **Emax (nm)** when the displayed x-axis is wavelength.
  These fields default to the uploaded spectrum's wavelength range. Changing
  them updates the displayed plot and publication PNG.

The publication PNG is exported at **300 dpi** with a **7 × 4.5 inch** figure
size. It omits the plot title and provisional-transcription note. Peak markers,
peak labels, and the legend are included only when detected peaks are enabled.

#### Available colormaps

| Category | Colormaps |
| --- | --- |
| Emission spectra | Inferno, Magma, Plasma, Viridis, Turbo, Hot, Cividis |
| Scientific comparison | Coolwarm, Seismic, RdYlBu, Spectral |
| Intensity maps | YlGnBu, YlOrRd, Purples, Ocean, Terrain, Gist Earth |
| Specialized | Rainbow, HSV, Cool, Twilight, Twilight Shifted |

**Inferno** is the default and, along with Magma, is especially suitable for
intensity variation on dark backgrounds. Diverging maps are generally better
for values centered around a meaningful midpoint, such as signed residuals.
Rainbow and HSV can create misleading visual boundaries; Twilight maps are
intended for cyclic data.

### Downloadable results

The application provides downloads for:

- Detected peak table (CSV).
- Peak-to-transition candidate matches (CSV).
- Reference transition catalog (CSV).
- Uploaded spectrum with converted energy coordinates (CSV).
- Publication-ready annotated spectrum (300 dpi PNG).

## Try the hosted application

Open [RareEarthsLab on Streamlit](https://rareearthslab-jxhqdrpyr3mtukkjbdaugf.streamlit.app/)
to analyze spectra in a browser without setting up a local Python environment.

### Basic workflow

1. Upload a two-column TXT, CSV, or DAT spectrum.
2. Select the spectral coordinate or use automatic detection.
3. Set the smoothing and peak-detection parameters.
4. Choose an ion and configure the transition-matching tolerance.
5. Inspect the detected peaks, candidate matches, and annotated spectrum.
6. Adjust plot appearance and, for wavelength plots, set Emin and Emax.
7. Download the CSV results and publication-ready PNG.

## Run locally

Requirements: Python and pip.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The application opens in a browser. If PowerShell blocks virtual-environment
activation, run the commands in a shell where activation is permitted, or
invoke the environment's Python directly with `.\.venv\Scripts\python.exe`.

## Technology

- [Python](https://www.python.org/) — application language.
- [Streamlit](https://streamlit.io/) — web interface.
- [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) — numeric
  operations and data processing.
- [SciPy](https://scipy.org/) — Savitzky–Golay smoothing and peak detection.
- [Matplotlib](https://matplotlib.org/) — spectrum plotting and PNG export.

## Reference data and scientific use

The included reference database cites energy-level tables by Carnall, Goodman,
Rajnak, and Rana (1989). Database coverage is partial; some entries are
provisional transcriptions or inferred labels. Consult the original tables and
verify each assignment before relying on it in a publication or scientific
interpretation.

RareEarthsLab is an analysis aid. Its output depends on the quality and
calibration of the input spectrum, user-selected analysis settings, and
reference-data coverage. Automated matches should be independently reviewed.

## License

RareEarthsLab is distributed under the [MIT License](LICENSE).
