#!/usr/bin/env python3
"""
Validate classic and modern pipelines against Mathcad RTF reference values.

Runs both pipelines on all PRN files with MCD parameters,
then compares results with extracted Mathcad values.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from propis_app.core.prn_reader import read_prn
from propis_app.core.pipeline import run_classic, run_modern, CycleParams, PipelineResult
from propis_app.core.rtf_parser import parse_all_rtf, MathcadResults


# MCD parameters extracted earlier (from binary MCD files)
# Format: (prn_file, mcd_name, n1, n2, im, isat, im1, isat1, tn)
# im, isat are RELATIVE to n1
# tn = saturation temperature (manually set per solution/day in Mathcad)
MCD_PARAMS = [
    # 020326 KDP нейтр XXXIII п.15 (tn=48.60)
    ("020326 KDP нейтр XXXIII п.15/__020326_2.prn",
     "__020326_2_1", 1, 8000, 5963, 6692, 5120, 7580, 48.60),
    ("020326 KDP нейтр XXXIII п.15/__020326_2.prn",
     "__020326_2_2", 13000, 20000, 4686, 5346, 3720, 6180, 48.60),
    # 020326 KDP нейтр XXXIII п.17 (tn=48.60)
    ("020326 KDP нейтр XXXIII п.17/__020326_5.prn",
     "__020326_5_1", 1, 9000, 7227, 7576, 6350, 8090, 48.60),
    ("020326 KDP нейтр XXXIII п.17/__020326_5.prn",
     "__020326_5_2", 13000, 21000, 5890, 6187, 5210, 6610, 48.60),
    # 030326 KDP нейтр XXXIII п.18 (tn unknown from RTF, use None=te)
    ("030326 KDP нейтр XXXIII п.18/__030326_2.prn",
     "__030326_2_2", 13500, 21000, 4596, 5082, 3500, 5690, None),
    ("030326 KDP нейтр XXXIII п.18/__030326_2.prn",
     "__030326_2_3", 25000, 32000, 4722, 5164, 3795, 5575, None),
    # 030326 KDP нейтр XXXIII п.26 (tn unknown from RTF, use None=te)
    ("030326 KDP нейтр XXXIII п.26/__030326_5.prn",
     "__030326_5_1", 1, 10000, 7606, 8193, 6570, 8990, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_5.prn",
     "__030326_5_2", 13000, 22000, 6260, 6824, 5080, 7500, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_6.prn",
     "__030326_6_1", 1, 10000, 6831, 7395, 5930, 7940, None),
    ("030326 KDP нейтр XXXIII п.26/__030326_6.prn",
     "__030326_6_3", 25000, 33000, 5200, 5758, 4240, 6330, None),
    # 040326 KDP нейтр XXXIII п.24 (tn=49.60 confirmed from RTF)
    ("040326 KDP нейтр XXXIII п.24/__040326_2.prn",
     "__040326_2_2", 13000, 20000, 5405, 6045, 4660, 6480, 49.60),
    ("040326 KDP нейтр XXXIII п.24/__040326_2.prn",
     "__040326_2_3", 26000, 32000, 4121, 4606, 3220, 5160, 49.60),
    # 040326 KDP нейтр XXXIV п.28 (DIFFERENT solution! tn unknown)
    ("040326 KDP нейтр XXXIV п.28/__040326_5.prn",
     "__040326_5_2", 12880, 22000, 6497, 7038, 5515, 7530, None),
    ("040326 KDP нейтр XXXIV п.28/__040326_5.prn",
     "__040326_5_3", 25000, 33000, 5926, 6468, 5070, 6880, None),
    ("040326 KDP нейтр XXXIV п.28/__040326_6.prn",
     "__040326_6_2", 12880, 22000, 6619, 7190, 5665, 7700, None),
]


def find_rtf_match(mcd_name: str, rtf_results: list[MathcadResults]) -> MathcadResults | None:
    """Find matching RTF result by MCD name."""
    for r in rtf_results:
        if mcd_name in r.filename:
            return r
    return None


def format_val(val, fmt=".2f"):
    """Format value or return '---'."""
    if val is None or val == 0.0:
        return "---"
    return f"{val:{fmt}}"


def run_validation():
    """Run both pipelines and compare with Mathcad."""
    base = project_root

    # Load RTF reference values
    print("Loading Mathcad RTF reference values...")
    rtf_results = parse_all_rtf(base)
    print(f"  Found {len(rtf_results)} RTF files\n")

    # Results storage
    all_classic = []
    all_modern = []
    all_mathcad = []

    print("=" * 120)
    print(f"{'File':<25} {'Mode':<8} {'te':>6} {'tn':>6} {'Td':>5} "
          f"{'Sigm':>6} {'s2':>5} {'Sig035':>6} {'s0':>6} {'N_pts':>5}")
    print("=" * 120)

    for prn_rel, mcd_name, n1, n2, im, isat, im1, isat1, tn in MCD_PARAMS:
        prn_path = base / prn_rel
        if not prn_path.exists():
            print(f"  SKIP: {prn_path} not found")
            continue

        # Read PRN
        try:
            prn = read_prn(prn_path)
        except Exception as e:
            print(f"  ERROR reading {prn_path}: {e}")
            continue

        params = CycleParams(n1=n1, n2=n2, im=im, isat=isat, im1=im1, isat1=isat1)

        # Run classic (with manual tn)
        try:
            classic = run_classic(prn, params, salt=1, acid=0, face=0,
                                  channel=1, dtau=0.055, tn_manual=tn)
            all_classic.append(classic)
        except Exception as e:
            print(f"  CLASSIC ERROR {mcd_name}: {e}")
            classic = None

        # Run modern
        try:
            modern = run_modern(prn, params, salt=1, acid=0, face=0,
                                channel=1, dtau=0.055)
            all_modern.append(modern)
        except Exception as e:
            print(f"  MODERN ERROR {mcd_name}: {e}")
            modern = None

        # Find Mathcad reference
        rtf = find_rtf_match(mcd_name, rtf_results)
        if rtf:
            all_mathcad.append(rtf)

        # Print results
        short_name = mcd_name.replace("__", "")

        def print_row(label, r):
            if r is None:
                print(f"{short_name:<25} {label:<8} {'--- no result ---'}")
                return
            if isinstance(r, PipelineResult):
                n_pts = len(r.fit_result.sigma_percent) if r.fit_result else 0
                print(f"{short_name:<25} {label:<8} "
                      f"{format_val(r.te):>6} {format_val(r.tn):>6} "
                      f"{format_val(r.Td):>5} {format_val(r.Sigm):>6} "
                      f"{format_val(r.s2):>5} {format_val(r.Sig035):>6} "
                      f"{format_val(r.s0):>6} {n_pts:>5}")
            elif isinstance(r, MathcadResults):
                print(f"{short_name:<25} {label:<8} "
                      f"{format_val(r.te):>6} {format_val(r.tn):>6} "
                      f"{format_val(r.Td):>5} {format_val(r.Sigm):>6} "
                      f"{format_val(r.s2):>5} {format_val(r.Sig035, '.1f'):>6} "
                      f"{'':>6} {'':>5}")

        print_row("MATHCAD", rtf)
        print_row("CLASSIC", classic)
        print_row("MODERN", modern)
        print("-" * 120)

    # Statistics
    print("\n\n" + "=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)

    _print_statistics("Classic vs Mathcad", all_classic, all_mathcad)
    _print_statistics("Modern vs Mathcad", all_modern, all_mathcad)
    _print_statistics("Modern vs Classic", all_modern, all_classic)


def _print_statistics(title: str, results_a, results_b):
    """Print comparison statistics between two result sets."""
    print(f"\n--- {title} ---")

    fields = ['te', 'tn', 'Td', 'Sigm', 's2', 'Sig035']
    diffs = {f: [] for f in fields}

    n_compared = min(len(results_a), len(results_b))
    for i in range(n_compared):
        a = results_a[i]
        b = results_b[i]

        for f in fields:
            va = getattr(a, f, None)
            vb = getattr(b, f, None)
            if va and vb and va != 0 and vb != 0:
                diffs[f].append(va - vb)

    print(f"{'Param':<8} {'N':>4} {'Mean diff':>10} {'Std diff':>10} {'Max |diff|':>10}")
    for f in fields:
        d = diffs[f]
        if d:
            d = np.array(d)
            print(f"{f:<8} {len(d):>4} {np.mean(d):>10.4f} {np.std(d):>10.4f} "
                  f"{np.max(np.abs(d)):>10.4f}")
        else:
            print(f"{f:<8} {'---':>4}")


if __name__ == "__main__":
    run_validation()
