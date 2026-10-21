# LuminescenceLab

**A free web application for rare earth emission spectroscopy and peak /
electronic transition identification.**

[![Launch LuminescenceLab](https://img.shields.io/badge/Launch-LuminescenceLab-168B8B?style=for-the-badge)](https://rareearthslab-jxhqdrpyr3mtukkjbdaugf.streamlit.app/)

LuminescenceLab helps researchers inspect experimental emission spectra, detect
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
  semicolon-separated input is supported. Non-numeric instrument metadata
  before the spectrum data is ignored.
- Use wavelength (nm), wavenumber (cm⁻¹), or photon energy (eV) as the spectral
  coordinate, or let the application infer the axis from the data.
- Smooth the signal with a Savitzky–Golay filter before detecting peaks.
- Tune peak prominence, minimum point separation, smoothing window, and
  optional minimum intensity.
- Convert input positions to wavelength and photon wavenumber for comparison.
- Upload multiple files to compare scans from the same sample. Choose an
  overlay or vertically stacked plot; each scan keeps its own measured
  intensities, with no averaging, combining, normalization, or interpolation.
  All scans must use the same x-axis coordinate; choose it explicitly if
  automatic detection assigns different units. Comparison controls let you
  switch between dark and white backgrounds, edit each scan's label, hide the
  scan labels/overlay legend, set E_min and
  E_max and Intensity_min and Intensity_max, and adjust the shared x/y
  axis-label font size, shared x/y tick font size, and scan-label/legend font
  size and spectrum line width. White-background plots use a high-contrast
  line palette. An optional intensity-background colormap can be enabled with
  the same colormap choices as the single-spectrum plot.
- Set individual peak integration limits in the editable table. Limits default
  to the detected peak's FWHM boundaries. The table displays each peak ID,
  peak position, editable integration limits, and the trapezoidal area under
  the raw signal between those limits. The area is not baseline-corrected and
  has units of intensity multiplied by the selected x-axis units. Detected
  peaks and transition candidates are available as CSV downloads rather than
  additional on-screen tables.
- For single-spectrum analysis, calculate the Eu³⁺ Judd–Ofelt parameters
  Ω₂, Ω₄, and Ω₆ below the integration table. Transition wavelengths,
  integrated peak areas, refractive index, and lifetime are editable and start
  with the defaults from the supplied JO calculation script. Results show the
  scaled Ω values (Ω × 10²⁰), their unscaled values, and all other quantities
  calculated by that script.

### Candidate transition matching

- Compare peak photon energies with transitions derived from the available
  reference energy levels.
- Match using a wavelength tolerance or an optional transition-energy
  tolerance (cm⁻¹).
- Select an ion, infer an ion from the uploaded filename where possible, or
  search across the available ions.
- Download candidate states, reference wavelengths, energy differences, and
  available source and transcription details in the transition-match CSV.

The quality and coverage of matches depend on experimental calibration,
peak-detection settings, and reference-data completeness.

### Plot customization

Use the sidebar to adjust:

- Dark or white plot background, plus an optional intensity-background
  colormap. White-background spectra use contrasting blue/orange lines with
  dark, readable peak and transition labels. Peak annotations and plot legends
  have transparent backgrounds and use foreground colors matched to the
  selected plot background.
- Peak-label font size, axis-label font size, and tick-label font size.
- Line width shared by the experimental and smoothed signals.
- Smoothed-signal color.
- Visibility of detected peak markers and their transition/position labels.
- The intensity-background colormap.
- **E_min (nm)** and **E_max (nm)** when the displayed x-axis is wavelength.
  These fields default to the uploaded spectrum's wavelength range. Changing
  them updates the displayed plot and publication PNG.
- **Intensity_min** and **Intensity_max**, which default to the spectrum's
  minimum and maximum counts. Changing them updates the displayed plot and
  publication PNG.

The single-spectrum publication PNG is exported at **300 dpi** with a
**7 × 4.5 inch** figure size. It omits the plot title and provisional-
transcription note. Peak markers, peak labels, and the legend are included
only when detected peaks are enabled.

Multi-scan comparison plots are also exported at **300 dpi** and **7 inches
wide** (double-column manuscript width). Overlay plots are **4.5 inches high**;
stacked plots are at least **4.5 inches high** and allocate **2.25 inches per
scan** when more height is needed. When the intensity colormap is enabled, its
colorbar retains its scale but has no axis label.

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

Open [LuminescenceLab on Streamlit](https://rareearthslab-jxhqdrpyr3mtukkjbdaugf.streamlit.app/)
to analyze spectra in a browser without setting up a local Python environment.

### Basic workflow

1. Upload one two-column TXT, CSV, or DAT spectrum for peak analysis, or upload
   multiple files to compare scans in overlay or stacked layout.
2. Select the spectral coordinate or use automatic detection.
3. Set the smoothing and peak-detection parameters.
4. Choose an ion and configure the transition-matching tolerance.
5. Edit per-peak integration limits if needed, then inspect the integrated
   areas and annotated spectrum. For a single spectrum, edit the Judd–Ofelt
   input defaults below the table and inspect Ω₂, Ω₄, and Ω₆.
6. Adjust plot appearance and, for wavelength plots, set E_min and E_max.
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

### Recommended citation

If you use LuminescenceLab in your research, please cite:

Khan, L. U. et al. (2026). “Lighting up materials with X-rays: Advanced X-ray
excited optical luminescence probe at the BM08-XAFS/XRF beamline, SESAME.”
*IUCrJ*, **13**. [https://journals.iucr.org/m/issues/2026/06/00/ro5048/index.html](https://journals.iucr.org/m/issues/2026/06/00/ro5048/index.html)

For work involving rare-earth complexes and functionalized Fe₃O₄
nanomarkers, also cite:

Khan, L. U.; Brito, H. F.; Hölsä, J.; Pirota, K. R.; Muraca, D.; Felinto,
M. C. F. C.; Teotonio, E. E. S.; Malta, O. L. (2014). “Red-green emitting
and superparamagnetic nanomarkers containing Fe₃O₄ functionalized with
calixarene and rare earth complexes.” *Inorganic Chemistry*, **53**(24),
12902–12910. [https://doi.org/10.1021/ic5018856](https://doi.org/10.1021/ic5018856)

### Reference database

The included reference database cites energy-level tables by Carnall, Goodman,
Rajnak, and Rana (1989). Database coverage is partial; some entries are
provisional transcriptions or inferred labels. Consult the original tables and
verify each assignment before relying on it in a publication or scientific
interpretation.

LuminescenceLab is an analysis aid. Its output depends on the quality and
calibration of the input spectrum, user-selected analysis settings, and
reference-data coverage. Automated matches should be independently reviewed.

## License

LuminescenceLab is distributed under the [MIT License](LICENSE).
