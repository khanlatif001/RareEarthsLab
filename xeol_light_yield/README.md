# Absolute XEOL light yield

This standalone tool estimates absolute light yield in photons per keV deposited
from an OceanView spectrum and a QEPro complete-chain absolute radiometric
calibration. It does not change or depend on the LuminescenceLab Streamlit app.
It adapts the absolute-yield principle of de Haas, Dorenbos, and van Eijk
(2005), *Measuring the absolute light yield of scintillators*. Their
PMT-specific quantum-efficiency and reflectivity factors are not applied to the
QEPro.

## Required calibration input

The QEPro certificate has been described as **wavelength calibration only**.
It relates detector pixels to wavelengths, but does not calibrate detector
signal to optical power; it cannot be used as the absolute-response table or
produce absolute photon numbers. The certificate file itself has not been
attached to the project. Obtain a separate traceable absolute radiometric
calibration for the complete measurement chain, then convert its response into
a CSV with these exact columns:

```csv
wavelength_nm,watts_per_nm_per_count_rate
350,1.2e-9
351,1.2e-9
```

Each response value must be **W/nm per count/s**, applicable to the full optical
chain used for the sample (including the same fiber, optics, windows, and
spectrometer settings). The example numbers are placeholders, not QEPro data.
The response table must cover the entire measured emission spectrum; no
extrapolation is performed. The wavelength-only certificate ID, if available,
can be entered separately for traceability. Its wavelength accuracy/range and
validity can be reviewed when the certificate file is attached; those facts do
not replace the radiometric calibration.

Provide collection efficiency separately, either as a scalar CLI value between
0 and 1 or as a wavelength-dependent CSV:

```csv
wavelength_nm,collection_efficiency
350,0.02
700,0.015
```

This is the fraction of the sample's total emitted photons represented by the
optical power reaching the calibrated measurement chain. Do not include the
same collection loss in both the spectral response and the efficiency factor.

## Calculation and limitations

The script subtracts an optional matching background, converts the signal to
spectral power, integrates photon rate using \(hc/\lambda\), corrects for
wavelength-dependent collection efficiency, and divides total emitted photons
by estimated deposited X-ray energy. The deposited-energy fraction must be
supplied by the user; it is not inferred from primary-beam attenuation.

Use `counts_per_integration` for counts per QEPro integration (the default), and
provide the integration time. Use `counts_per_second` if the export is already
in count-rate units. Independently provide the **actual beam-on duration**.
Integration time multiplied by number of scans is not assumed to equal this
duration.

The optional combined standard uncertainty is calculated only when all five
relative standard uncertainties are supplied: spectral response, collection
efficiency, incident flux, X-ray energy, and deposited fraction. Exposure
duration is shared by emitted-photon and deposited-energy totals, so it cancels
from the yield ratio. The uncertainty combines the remaining inputs as
independent and does not add counting statistics unless those are included in
the supplied uncertainty values.

## Inputs and command

The spectrum is a two-column TXT, CSV, or DAT file: wavelength in nm and QEPro
signal. Leading instrument metadata and a text header are allowed. If a
background is supplied, it must cover the sample spectrum's full wavelength
range and use matching acquisition settings.

```powershell
python xeol_light_yield\analyze_xeol_light_yield.py `
  --spectrum sample.txt `
  --response qepro_absolute_response.csv `
  --collection-efficiency 0.02 `
  --signal-basis counts_per_integration `
  --integration-time-s 5 `
  --exposure-time-s 5 `
  --incident-flux-photons-s 1e8 `
  --xray-energy-kev 8.1 `
  --deposited-fraction 0.5 `
  --sample-composition "GGG:Eu 5%" `
  --sample-thickness-mm 0.5 `
  --collection-geometry "describe the sample, fiber, and optical path" `
  --radiometric-calibration-identifier "absolute response calibration ID" `
  --wavelength-calibration-identifier "QEPro wavelength certificate ID"
```

All command-line measurement values above are examples only. The script writes a
JSON report containing the result, wavelength range, both calibration
identifiers, acquisition details, sample and geometry metadata, deposited-energy
estimate, assumptions, and uncertainty status.

Run the focused tests from the repository root:

```powershell
python -m unittest discover -s xeol_light_yield -v
```
