#!/usr/bin/env python3
"""Python translation of the equations in 'Mathcad-Omegas NOVOS.xmcd'.

Edit the input values in the INPUT PARAMETERS section, then run:
    python mathcad_omegas_to_python.py

The formulas and numerical constants below are transcribed from the supplied
Mathcad 14 worksheet. Check the scientific model/constants and units against
the original worksheet/reference before using results in a publication.
"""
from dataclasses import dataclass, asdict


@dataclass
class Inputs:
    # Transition wavelengths in nm
    Bar01nm: float = 592.8
    Bar02nm: float = 612.1
    Bar04nm: float = 700.9
    Bar06nm: float = 592.57

    # S parameters from the Mathcad worksheet
    S00: float = 5731.1925
    S01: float = 54247.4045
    S02: float = 442309.545
    S04: float = 167702.782
    S06: float = 0.0

    # Refractive index and lifetime
    n: float = 1.5
    tau_s: float = 0.23e-3  # seconds; Mathcad τ = 0.34 × 10^-3


def calculate(p: Inputs) -> dict[str, float]:
    """Calculate the quantities defined in the Mathcad worksheet."""
    # Convert wavelength (nm) to wavenumber (cm^-1): sigma = 1 / wavelength(cm)
    sigma01 = 1.0 / (p.Bar01nm * 1e-7)
    sigma02 = 1.0 / (p.Bar02nm * 1e-7)
    sigma04 = 1.0 / (p.Bar04nm * 1e-7)
    sigma06 = 1.0 / (p.Bar06nm * 1e-7)

    # Radiative transition probability and refractive-index correction
    A01 = 0.31e-11 * (p.n**3) * (sigma01**3)
    chi = p.n * (p.n**2 + 2)**2 / 9.0

    # Other radiative transition probabilities
    A02 = A01 * (p.S02 / p.S01)
    A04 = (p.S04 / p.S01) * A01
    A06 = (p.S06 / p.S01) * A01

    # Omega parameters (as defined in the worksheet)
    Omega2 = A02 / ((2.33e8 * sigma02**3) * chi)
    Omega4 = A04 / ((7.23e10 * sigma04**3) * chi * 0.0023)
    Omega6 = A06 / ((7.23e10 * sigma06**3) * chi * 0.0002)

    # Scaled Omega parameters
    rho2 = Omega2 * 1e20
    rho4 = Omega4 * 1e20
    rho6 = Omega6 * 1e20

    R02 = p.S00 / p.S02
    Arad = A01 + A02 + A04
    Atotal = 1.0 / p.tau_s
    Anr = Atotal - Arad
    eta = (Arad * 100.0) / Atotal
    nH2O = 1.05 * ((Atotal - Arad) / 1000.0)

    return {
        "sigma01_cm^-1": sigma01,
        "sigma02_cm^-1": sigma02,
        "sigma04_cm^-1": sigma04,
        "sigma06_cm^-1": sigma06,
        "A01": A01,
        "chi": chi,
        "A02": A02,
        "A04": A04,
        "A06": A06,
        "Omega2": Omega2,
        "Omega4": Omega4,
        "Omega6": Omega6,
        "rho2": rho2,
        "rho4": rho4,
        "rho6": rho6,
        "R02": R02,
        "Arad": Arad,
        "Atotal": Atotal,
        "Anr": Anr,
        "eta_percent": eta,
        "nH2O": nH2O,
    }


def main() -> None:
    inputs = Inputs()
    results = calculate(inputs)

    print("Mathcad-Omegas NOVOS — Python translation")
    print("=" * 58)
    print("INPUT PARAMETERS")
    for name, value in asdict(inputs).items():
        print(f"  {name:12s} = {value:.10g}")

    print("\nCALCULATED RESULTS")
    for name, value in results.items():
        print(f"  {name:18s} = {value:.10g}")

    print("\nNotes:")
    print("  • Wavelengths are converted from nm to wavenumbers in cm^-1.")
    print("  • eta_percent is the worksheet's eta value expressed as percent.")
    print("  • S06 = 0 in the supplied worksheet, so A06, Omega6 and rho6 are zero.")
    print("  • The worksheet defines Arad as A01 + A02 + A04 (not including A06).")
    print("  • Verify definitions, constants, and units against the original model.")


if __name__ == "__main__":
    main()
