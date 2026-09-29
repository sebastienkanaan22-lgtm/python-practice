"""Plot a wavelength/power export and estimate its spectral FWHM.

Usage: python spectrum_fwhm.py 0freq.txt --output spectrum_fwhm.png
Requires: numpy and matplotlib.

Assumptions: wavelengths are in metres and power is on a 10*log10(power)
scale (dB or dBm). The supplied file has no unit labels. FWHM is evaluated
in wavelength, using linear interpolation of normalized linear power.
No smoothing, fitting, or background subtraction is applied.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def read_spectrum(path):
    """Read numeric tab-delimited pairs, ignoring binary export metadata."""
    raw = Path(path).read_bytes()
    number = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    pattern = rb"(" + number + rb")\t(" + number + rb")[ \t]*(?:\r?\n|$)"
    matches = list(re.finditer(pattern, raw))
    pairs = np.array([(float(m[1]), float(m[2])) for m in matches])
    if len(pairs) < 3 or not np.isfinite(pairs).all():
        raise ValueError("The file must contain at least three finite numeric pairs.")

    # This export repeats its 510 wavelengths. Keep the first reading only
    # after checking that repeated powers agree to numerical precision.
    wavelength_m, first, inverse = np.unique(
        pairs[:, 0], return_index=True, return_inverse=True
    )
    power_db = pairs[first, 1]
    max_duplicate_difference = float(np.max(np.abs(pairs[:, 1] - power_db[inverse])))
    if max_duplicate_difference > 1e-9:
        raise ValueError("Repeated wavelengths have different powers. Select a single trace.")
    if np.any(wavelength_m <= 0):
        raise ValueError("Wavelengths must be positive.")
    metadata = {
        "parsed_rows": len(pairs),
        "unique_wavelengths": len(wavelength_m),
        "duplicate_rows_removed": len(pairs) - len(wavelength_m),
        "maximum_duplicate_difference_db": max_duplicate_difference,
    }
    return wavelength_m * 1e9, power_db, metadata


def calculate_fwhm(wavelength_nm, power_db):
    peak_index = int(np.argmax(power_db))
    peak_db = float(power_db[peak_index])
    normalized = 10.0 ** ((power_db - peak_db) / 10.0)
    # A 50% linear-power level is 3.0102999566 dB below the maximum.
    half_db = peak_db - 10.0 * np.log10(2.0)
    pairs = np.flatnonzero((normalized[:-1] >= 0.5) != (normalized[1:] >= 0.5))
    left_candidates = pairs[pairs < peak_index]
    right_candidates = pairs[pairs >= peak_index]
    if not len(left_candidates) or not len(right_candidates):
        raise ValueError("The peak is not bounded by half-power crossings on both sides.")
    left_pair = int(left_candidates[-1])
    right_pair = int(right_candidates[0])

    def crossing(i):
        return float(wavelength_nm[i] + (0.5 - normalized[i]) *
                     (wavelength_nm[i + 1] - wavelength_nm[i]) /
                     (normalized[i + 1] - normalized[i]))

    left_nm = crossing(left_pair)
    right_nm = crossing(right_pair)
    width_nm = right_nm - left_nm
    result = {
        "wavelength_unit_assumed": "metres in input, nanometres in plot",
        "power_scale_assumed": "dB power scale (absolute reference unspecified)",
        "method": "linear interpolation of normalized linear power, no smoothing or fit",
        "peak_wavelength_nm": float(wavelength_nm[peak_index]),
        "peak_power_db": peak_db,
        "half_power_level_db": float(half_db),
        "left_half_max_nm": left_nm,
        "right_half_max_nm": right_nm,
        "spectral_fwhm_nm": width_nm,
        "spectral_fwhm_pm": width_nm * 1000.0,
        "frequency_span_between_crossings_GHz": float(
            299792458.0 * (1.0 / left_nm - 1.0 / right_nm)
        ),
        "half_power_crossing_count": len(pairs),
        "sample_spacing_near_peak_nm": float(
            (wavelength_nm[peak_index + 1] - wavelength_nm[peak_index - 1]) / 2.0
        ),
    }
    return normalized, result


def plot_spectrum(wavelength_nm, power_db, normalized, result, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8D98A7", "axes.labelcolor": "#263547",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "savefig.facecolor": "white",
    })
    fig, (full, zoom) = plt.subplots(
        1, 2, figsize=(12.6, 6.0), gridspec_kw={"width_ratios": [1, 1.28]}
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.21, top=0.77, wspace=0.29)
    fig.text(0.075, 0.945, "Spectrum and spectral FWHM", fontsize=20,
             fontweight="bold", color="#142C45")
    fig.text(0.075, 0.887,
             f"FWHM = {result['spectral_fwhm_nm']:.4f} nm"
             f"  ({result['spectral_fwhm_pm']:.1f} pm)"
             f"     |     Peak wavelength = {result['peak_wavelength_nm']:.4f} nm",
             fontsize=12.5, color="#245C83")

    full.plot(wavelength_nm, power_db, color="#196C99", lw=1.8)
    full.axhline(result["half_power_level_db"], color="#BC541F", lw=1.2, ls="--",
                 label=f"Half power: {result['half_power_level_db']:.4f} dB")
    full.scatter([result["peak_wavelength_nm"]], [result["peak_power_db"]],
                 color="#196C99", s=27, zorder=4)
    full.set(title="Full recorded spectrum", xlabel="Wavelength (nm)",
             ylabel="Power (dB, as recorded)")
    full.set_ylim(min(power_db) - 4, max(power_db) + 9)
    full.set_xlim(wavelength_nm[0], wavelength_nm[-1])
    full.xaxis.set_major_locator(MaxNLocator(5))
    full.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    full.text(0.975, result["half_power_level_db"] + 2.5,
              f"Half power: {result['half_power_level_db']:.4f} dB",
              transform=full.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=9, color="#A44317")

    left = result["left_half_max_nm"]
    right = result["right_half_max_nm"]
    peak = result["peak_wavelength_nm"]
    width = result["spectral_fwhm_nm"]
    zoom.plot(wavelength_nm, normalized, color="#196C99", lw=1.7,
              marker="o", markersize=3.5, label="Samples and linear interpolation")
    zoom.axhline(0.5, color="#BC541F", lw=1.1, ls="--")
    zoom.vlines([left, right], 0, 0.5, colors="#BC541F", linestyles=":", lw=1.2)
    zoom.scatter([left, right], [0.5, 0.5], s=45, color="#BC541F", zorder=5)
    zoom.annotate("", xy=(left, 0.5), xytext=(right, 0.5),
                  arrowprops={"arrowstyle": "<->", "color": "#BC541F", "lw": 1.4})
    zoom.text((left + right) / 2.0, 0.56, f"{width:.4f} nm", ha="center",
              va="bottom", fontweight="bold", color="#A44317",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    zoom.set(title="Peak detail in linear power", xlabel="Wavelength (nm)",
             ylabel="Normalized power", xlim=(peak - 1.75 * width, peak + 1.75 * width),
             ylim=(0, 1.09))
    zoom.xaxis.set_major_locator(MaxNLocator(5))
    zoom.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for axis in (full, zoom):
        axis.grid(True, color="#DFE5EC", lw=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    fig.text(0.075, 0.11,
             f"Half-maximum wavelengths: {left:.4f} nm and {right:.4f} nm.",
             fontsize=10.5, color="#334155")
    fig.text(0.075, 0.065,
             "Assumptions: wavelength in metres; power on a dB scale. "
             f"{len(wavelength_nm)} distinct wavelengths; repeated readings agree to numerical precision.",
             fontsize=9, color="#64748B")
    fig.text(0.075, 0.034,
             "FWHM uses interpolation in linear power, with no smoothing, fitting, or background subtraction.",
             fontsize=9, color="#64748B")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("spectrum_fwhm.png"))
    args = parser.parse_args()
    wavelength_nm, power_db, metadata = read_spectrum(args.input)
    normalized, result = calculate_fwhm(wavelength_nm, power_db)
    result.update(metadata)
    plot_spectrum(wavelength_nm, power_db, normalized, result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()