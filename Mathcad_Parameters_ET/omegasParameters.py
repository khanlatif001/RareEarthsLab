#!/usr/bin/env python3
"""Judd-Ofelt (JO) parameters (Ω₂, Ω₄, and Ω₆) calculation for Eu³⁺ in Europium compounds.
"""
from dataclasses import dataclass, asdict
import sys


@dataclass
class Inputs:
    # Electronic Transition Position (wavelengths in nm)
    Transition5D0_7F1: float = 592.8
    Transition5D0_7F2: float = 612.1
    Transition5D0_7F4: float = 700.9
    Transition5D0_7F6: float = 810.0

    # Integrated area of each transition peak (S)
    S5D0_7F0: float = 5731.1925
    S5D0_7F1: float = 54247.4045
    S5D0_7F2: float = 442309.545
    S5D0_7F4: float = 167702.782
    S5D0_7F6: float = 0.0

    # Refractive index and lifetime
    n: float = 1.5
    tau_s: float = 0.23e-3  # seconds; Mathcad τ = 0.34 × 10^-3


def calculate(p: Inputs) -> dict[str, float]:
    """Calculate the quantities defined in the Mathcad worksheet."""
    # Convert wavelength (nm) to wavenumber (cm^-1): sigma = 1 / wavelength(cm)
    sigma01 = 1.0 / (p.Transition5D0_7F1 * 1e-7)
    sigma02 = 1.0 / (p.Transition5D0_7F2 * 1e-7)
    sigma04 = 1.0 / (p.Transition5D0_7F4 * 1e-7)
    sigma06 = 1.0 / (p.Transition5D0_7F6 * 1e-7)

    # Radiative transition probability and refractive-index correction
    A01 = 0.31e-11 * (p.n**3) * (sigma01**3)
    chi = p.n * (p.n**2 + 2)**2 / 9.0

    # Other radiative transition probabilities
    A02 = A01 * (p.S5D0_7F2 / p.S5D0_7F1)
    A04 = (p.S5D0_7F4 / p.S5D0_7F1) * A01
    A06 = (p.S5D0_7F6 / p.S5D0_7F1) * A01

    # Omega parameters (as defined in the worksheet)
    Omega2 = A02 / ((2.33e8 * sigma02**3) * chi)
    Omega4 = A04 / ((7.23e10 * sigma04**3) * chi * 0.0023)
    Omega6 = A06 / ((7.23e10 * sigma06**3) * chi * 0.0002)

    # Scaled Omega parameters
    rho2 = Omega2 * 1e20
    rho4 = Omega4 * 1e20
    rho6 = Omega6 * 1e20

    R02 = p.S5D0_7F0 / p.S5D0_7F2
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    inputs = Inputs()
    results = calculate(inputs)

    print("Mathcad-Omegas NOVOS — Python translation")
    print("=" * 58)
    print("INPUT PARAMETERS")
    for name, value in asdict(inputs).items():
        display_name = {"tau_s": "τ (s)"}.get(name, name)
        print(f"  {display_name:12s} = {value:.10g}")

    print("\nCALCULATED RESULTS")
    for name, value in results.items():
        display_name = {
            "rho2": "Ω₂",
            "rho4": "Ω₄",
            "rho6": "Ω₆",
            "eta_percent": "Intrinsic Quantum Yield (%)",
        }.get(name, name)
        print(f"  {display_name:18s} = {value:.10g}")

    print("\nNotes:")
    print("  • Wavelengths are converted from nm to wavenumbers in cm^-1.")
    print("  • eta_percent is the worksheet's eta value expressed as percent.")
    print("  • S5D0_7F6 = 0 in the supplied worksheet, so A06, Omega6 and Ω₆ are zero.")
    print("  • The worksheet defines Arad as A01 + A02 + A04 (not including A06).")
    print("  • Verify definitions, constants, and units against the original model.")


if __name__ == "__main__":
    main()
